# The LLVM Compiler Infrastructure (forked by next)

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/llvm/llvm-project/badge)](https://securityscorecards.dev/viewer/?uri=github.com/llvm/llvm-project)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8273/badge)](https://www.bestpractices.dev/projects/8273)
[![libc++](https://github.com/llvm/llvm-project/actions/workflows/libcxx-build-and-test.yaml/badge.svg?branch=main&event=schedule)](https://github.com/llvm/llvm-project/actions/workflows/libcxx-build-and-test.yaml?query=event%3Aschedule)

Welcome to the LLVM project!

This repository contains the source code for LLVM, a toolkit for the
construction of highly optimized compilers, optimizers, and run-time
environments.

The LLVM project has multiple components. The core of the project is
itself called "LLVM". This contains all of the tools, libraries, and header
files needed to process intermediate representations and convert them into
object files. Tools include an assembler, disassembler, bitcode analyzer, and
bitcode optimizer.

C-like languages use the [Clang](https://clang.llvm.org/) frontend. This
component compiles C, C++, Objective-C, and Objective-C++ code into LLVM bitcode
-- and from there into object files, using LLVM.

Other components include:
the [libc++ C++ standard library](https://libcxx.llvm.org),
the [LLD linker](https://lld.llvm.org), and more.

## Getting the Source Code and Building LLVM

Consult the
[Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
page for information on building and running LLVM.

For information on how to contribute to the LLVM project, please take a look at
the [Contributing to LLVM](https://llvm.org/docs/Contributing.html) guide.

## Pull request merge flow

LLVM pull request merge flow auto bump the toolchain version.
When merging a pull request, a commit with an updated toolchain version is added to the pull request as part of LLVM pre merge flow.
`next-llvm-project` version is set at the toolchain [version file](https://github.com/nextsilicon/next-llvm-project/blob/next_release_170/nextsilicon/nextcc/CMakeLists.txt), no need to update the version manualy.
When a newer toolchain version is desired in `nextutils`, please set the expected toolchain version in `nextutils` expected toolchain [version file](https://github.com/nextsilicon/nextutils/blob/master/cmake/NextLLVMVersion.cmake).

Toolchain versioning levels - `MAJOR.MINOR.PATCHLEVEL` (`X.Y.Z`):
1. **MAJOR (X)** - major version update (usually requires code changes in `nextutils` repository).
2. **MINOR (Y)** - minor version update (usually requires code changes in `nextutils` repository).
3. **PATCHLEVEL (Z)** - patch level version update.

The pull request pre merge flow is triggered by a comment.
The comment set the toolchain bump level - `[ci-merge-<bump_level>]`:
1. `[ci-merge-major]` - trigger *pre merge* that bump toolchaion MAJOR version.
2. `[ci-merge-minor]` - trigger *pre merge* that bump toolchaion MINOR version.
3. `[ci-merge-patch]` - trigger *pre merge* that bump toolchaion PATCHLEVEL version.

In order to test pull request pre merge flow, trigger pre merge test by a comment.
The comment set the toolchain bump level - `[ci-test-<bump_level>]`:
1. `[ci-test-major]` - trigger *test pre merge* that bump toolchaion MAJOR version without merging the PR.
2. `[ci-test-minor]` - trigger *test pre merge* that bump toolchaion MINOR version without merging the PR.
3. `[ci-test-patch]` - trigger *test pre merge* that bump toolchaion PATCHLEVEL version without merging the PR.

## Getting in touch

Join the [LLVM Discourse forums](https://discourse.llvm.org/), [Discord
chat](https://discord.gg/xS7Z362),
[LLVM Office Hours](https://llvm.org/docs/GettingInvolved.html#office-hours) or
[Regular sync-ups](https://llvm.org/docs/GettingInvolved.html#online-sync-ups).

The LLVM project has adopted a [code of conduct](https://llvm.org/docs/CodeOfConduct.html) for
participants to all modes of communication within the project.
