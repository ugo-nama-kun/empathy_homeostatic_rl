# Homeostatic Couplings for Prosocial Behaviors
This is the experimental code for the homeostatic-regulation reinforcement learning (HRRL) with homeostatic coupling.


### Requirements
```text
torch
tyro
gymnasium
tensorboard
pygame

# gymnasium environments, available from: https://github.com/ugo-nama-kun/tiny_empathy
tiny_empathy==1.9.3
```

### How to Run
#### Experiments with direct observation of other's internal state
`--enable-empathy` option enable the cognitive empathy in the agent and environment setting, while `--weight-empathy` option sets
the strength of the affective empathy (default: 0.0, disabling the affective empathy).
```shell
python train_sharefood.py --enable-empathy --weight_empathy=0.5
```

#### Experiments with self-decoder experiments
In the self-decoder environments (foodshare & grid_rooms), cognitive empathy is enabled by default. `--enable-learning` option
toggle the learning process of the self-decoder. `--decoding-mode` takes two modes, `full` or `affect`. 

The `full` mode 
decodes physiological responses (emotional-feature) of the other agent, for both of the observation of HRRL and reward (drive) calculation.


The `affect` mode decode the physiological response only for the reward calculation and the HRRL use the physiological response as an observation directly.
```shell
python train_foodshare_decoder_learning.py --enable-learning --weight_empathy=0.5 --decoding-mode=full
```



### Citation
```text
@article{anonymous2025homeostatic,
  title={Homeostatic Couplings for Prosocial Behaviors},
  author={Anonymous Authors},
  year={2025},
  publisher={GitHub}
}
```


