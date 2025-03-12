# BOLT-based binary analysis

As part of post-link-time optimizing, BOLT needs to perform a range of analyses
on binaries such as recontructing control flow graphs, and more.

The `llvm-bolt-binary-analysis` tool enables running requested binary analyses
on binaries, and generating reports. It does this by building on top of the
analyses implemented in the BOLT libraries.

## Which binary analyses are implemented?

At the moment, no binary analyses are implemented.

The goal is to make it easy using a plug-in framework to add your own analyses.

## How to add your own binary analysis

_TODO: this section needs to be written. Ideally, we should have a simple
"example" or "template" analysis that can be the starting point for implementing
custom analyses_
