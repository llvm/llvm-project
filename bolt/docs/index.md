# BOLT

```{toctree}
:hidden: true

BOLTAArch64OptimizationStatus
```

BOLT is a post-link optimizer developed to speed up large applications.
It achieves the improvements by optimizing application’s code layout
based on execution profile gathered by sampling profiler, such as Linux
`perf` tool. An overview of the ideas implemented in BOLT along with a
discussion of its potential and current results is available in [CGO’19
paper](https://research.fb.com/publications/bolt-a-practical-binary-optimizer-for-data-centers-and-beyond/).

# User Guides

```{toctree}
:maxdepth: 1

GettingStarted
OptimizingClang
OptimizingLinux
Heatmaps
```

# Reference

```{toctree}
:maxdepth: 1

CommandLineArgumentReference
profiles
BOLTAArch64OptimizationStatus
```

# Design Documentation

```{toctree}
:maxdepth: 1

BAT
BinaryAnalysis
NewBackend
PointerAuthDesign
RuntimeLibrary
```
