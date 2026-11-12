# BOLT

BOLT is a post-link optimizer developed to speed up large applications.
It achieves the improvements by optimizing application’s code layout
based on execution profile gathered by sampling profiler, such as Linux
`perf` tool. An overview of the ideas implemented in BOLT along with a
discussion of its potential and current results is available in [CGO’19
paper](https://research.fb.com/publications/bolt-a-practical-binary-optimizer-for-data-centers-and-beyond/).

# User Guides

```{toctree}
:caption: User Guides
:maxdepth: 1

Getting Started <GettingStarted>
Optimizing Clang <OptimizingClang>
Optimizing Linux Kernel <OptimizingLinux>
Code Heatmaps <Heatmaps>
```

# Reference

```{toctree}
:caption: Reference
:maxdepth: 1

Command-Line Reference <CommandLineArgumentReference>
Profile Formats <profiles>
AArch64 Optimization Status <BOLTAArch64OptimizationStatus>
```

# Design Documentation

```{toctree}
:caption: Design Documentation
:maxdepth: 1

BOLT Address Translation <BAT>
Binary Analysis <BinaryAnalysis>
Adding a New Backend <NewBackend>
Pointer Authentication <PointerAuthDesign>
ORC-Based Linker <RuntimeLibrary>
```
