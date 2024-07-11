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


---
`Flang` 是一个基于 LLVM 的新 Fortran 编译器，它支持 Fortran 2018 以及一些扩展。这个编译器的设计、实现和开发参与方式都是值得关注的。
`Flang` 编译器的作用是将 Fortran 源代码转换成可执行文件。这个过程包括三个高级阶段：分析、降低（lowering）和代码生成/链接。在分析阶段，Fortran 源代码被转换成一个装饰过的解析树和符号表。
总的来说，`Flang` 作为 LLVM 项目的一部分，为 Fortran 语言提供了现代化的编译支持，使得 Fortran 程序能够在不同的平台上高效运行。

Fortan语言专注于科学计算，可能在量化中很有用。
