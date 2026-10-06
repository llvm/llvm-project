# LLVM Toolchain Build

This directory contains cache files for building a complete LLVM-based toolchain.
The resulting clang build will be LTO, PGO, and BOLT optimized and statically
linked against libc++ and compiler-rt.

The build is done in 3 stages:

* Stage 1: Build an LTO optimized libc++ with Stage 1 clang/lld.
* Stage 2 Instrumented: Build clang with instrumentation in order to generate
  profile data for PGO.
* Stage 2: Build clang with LTO, PGO, and BOLT optimizations and statically link
           with stage2 libc++ and compiler-rt.

## Usage

::
  $ cmake -S llvm -B build -C clang/cmake/caches/llvm-toolchain/stage1.cmake
  $ ninja stage2-install-distribution
