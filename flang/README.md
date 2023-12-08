# Flang

Flang is a ground-up implementation of a Fortran front end written in modern
C++. It started off as the f18 project (https://github.com/flang-compiler/f18)
with an aim to replace the previous flang project
(https://github.com/flang-compiler/flang) and address its various deficiencies.
F18 was subsequently accepted into the LLVM project and rechristened as Flang.

Please note that flang is not ready yet for production usage.

## Getting Started

Read more about flang in the [docs directory](docs).
Start with the [compiler overview](docs/Overview.md).

To better understand Fortran as a language
and the specific grammar accepted by flang,
read [Fortran For C Programmers](docs/FortranForCProgrammers.md)
and
flang's specifications of the [Fortran grammar](docs/f2018-grammar.md)
and
the [OpenMP grammar](docs/OpenMP-4.5-grammar.md).

Treatment of language extensions is covered
in [this document](docs/Extensions.md).

To understand the compilers handling of intrinsics,
see the [discussion of intrinsics](docs/Intrinsics.md).

To understand how a flang program communicates with libraries at runtime,
see the discussion of [runtime descriptors](docs/RuntimeDescriptor.md).

If you're interested in contributing to the compiler,
read the [style guide](docs/C++style.md)
and
also review [how flang uses modern C++ features](docs/C++17.md).

If you are interested in writing new documentation, follow
[LLVM's Markdown style guide](https://github.com/llvm/llvm-project/blob/main/llvm/docs/MarkdownQuickstartTemplate.md).

Consult the [Getting Started with Flang](docs/GettingStarted.md)
for information on building and running flang.
