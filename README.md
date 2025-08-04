# The LLVM Compiler Infrastructure

Welcome to the LLVM project!

This repository contains the source code for LLVM, a toolkit for the construction of highly optimized compilers, optimizers, and run-time environments.

The LLVM project has multiple components. The core of the project is itself called "LLVM." This contains all tools, libraries, and header files needed to process intermediate representations and convert them into object files. Tools include an assembler, disassembler, bitcode analyzer, and bitcode optimizer.

C-like languages use the Clang frontend. This component compiles C, C++, Objective-C, and Objective-C++ code into LLVM bitcode—and from there into object files, using LLVM.

Other project components include:
- the libc++ C++ standard library,
- the LLD linker,
- and more, such as LLDB (debugger), Flang (Fortran frontend), MLIR (multi-level IR), OpenMP runtime, Polly (polyhedral optimizer), and others.

## Getting the Source Code and Building LLVM

Consult the [Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm) page for the latest information on building and running LLVM.

For information on how to contribute to the LLVM project, please see the [Contributing to LLVM](https://llvm.org/docs/Contributing.html) guide.

## Releases

LLVM has regular, versioned releases. The latest release as of July 2025 is **LLVM 20.1.8**. See the [GitHub Releases](https://github.com/llvm/llvm-project/releases) page for release notes and download links.

## Getting in Touch

Join the [LLVM Discourse forums](https://discourse.llvm.org/), [Discord chat](https://discord.gg/xS7Z362), [LLVM Office Hours](https://llvm.org/docs/GettingInvolved.html#office-hours), or [regular online sync-ups](https://llvm.org/docs/GettingInvolved.html#online-sync-ups).

## Code of Conduct

The LLVM project has adopted a [code of conduct](https://llvm.org/docs/CodeOfConduct.html) which applies to all modes of communication within the project.

## About

The LLVM Project is a collection of modular and reusable compiler and toolchain technologies. For more information, visit [llvm.org](https://llvm.org/).
