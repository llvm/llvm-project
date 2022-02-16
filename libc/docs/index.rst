========================================================
llvm-libc: An ISO C-conformant Standard Library for LLVM
========================================================

**llvm-libc library is not complete.  If you need a fully functioning libc right
now, you should continue to use your standard system libc.**

.. contents:: Table of Contents
  :depth: 4
  :local:

Goals
=====

- C17 and upwards conformant.
- A modular libc with individual pieces implemented in the "as a
  library" philosophy of the LLVM project.
- Ability to layer this libc over the system libc if possible and desired
  for a platform.
- Provide POSIX extensions on POSIX compliant platforms.
- Provide system-specific extensions as appropriate. For example,
  provide the Linux API on Linux.
- Designed and developed from the start to work with LLVM tooling, fuzz testing
  and sanitizer-supported testing.
- Use source based implementations as far possible rather than
  assembly. Will try to *fix* the compiler rather than use assembly
  language workarounds.
- Extensive unit testing and standards conformance testing.

Why a new C Standard Library?
=============================

Implementing a libc is no small task and is not be taken lightly. A
natural question to ask is, "why a new implementation of the C
standard library?" Some of the major reasons are as follows:

- Rather than being built as a single monolithic codebase, llvm-libc is designed
  from the beginning to enable picking and choosing pieces.  This allows using
  it as a minimum overlay for e.g. faster math functions than might be
  available on the system library.  This is useful where an application may
  need to access improved CPU support over what's available on the system,
  or may need guarantees in performance across different installs.
- Explicit support for building llvm-libc and code with sanitizer compiler
  options.
- `Fuzzing`__
- Be useful for research and review.  By avoiding assembly language, using C++
  iterators, RAII and templates, llvm-libc aims to have clearly
  readable code and to improve the compiler as needed to ensure that optimal
  assembly is emitted.
- Enable fully static compiles.

.. __: https://github.com/llvm/llvm-project/tree/main/libc/fuzzing

Platform Support
================

Most development is currently targeting x86_64 and aarch64 on Linux.  Several
functions in llvm-libc have been tested on Windows.  The Fuchsia platform is
slowly replacing functions from its bundled libc with functions from llvm-libc.

ABI Compatibility
=================

llvm-libc is written to be ABI independent.  Interfaces are generated using
LLVM's tablegen, so supporting arbitrary ABIs is possible.  In it's initial
stages llvm-libc is not offering ABI stability in any form.

Other Interesting Documentation
===============================

.. toctree::

    build_system
    clang_tidy_checks
    entrypoints
    fuzzing
    ground_truth_specification
    header_generation
    implementation_standard
    integration_test
    layering
    mechanics_of_public_api
    redirectors
    source_layout
