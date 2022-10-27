==================
The LLVM C Library
==================

.. warning::
  The libc is not complete.  If you need a fully functioning C library right
  now, you should continue to use your standard system libraries.

Introduction
============

The libc aspires to a unique place in the software ecosystem.  The goals are:

- Fully compliant with current C standards (C17 and upcoming C2x) and POSIX.
- Easily decomposed and embedded: Supplement or replace system C library
  functionality easily.  This is useful to get consistent math precision across
  systems, or updated memory operations for newer microarchitectures.  These
  pieces will work on Linux, MacOS, Windows, and Fuchsia.
- The creation of fully static binaries without license implications.
- Increase whole program optimization opportunities for static binaries through
  ability to inline math and memory operations.
- Reduce coding errors by coding in modern C++ through the use of lightweight
  containers during coding that can be optimized away at runtime.
- Permit fuzzing and sanitizer instrumentation of user binaries including the
  libc functions.
- A complete testsuite that tests both the public interface and internal
  algorithms.
- `Fuzzing`__

.. __: https://github.com/llvm/llvm-project/tree/main/libc/fuzzing

Platform Support
================

Most development is currently targeting x86_64 and aarch64 on Linux.  Several
functions in the libc have been tested on Windows.  The Fuchsia platform is
slowly replacing functions from its bundled libc with functions from this
project.

ABI Compatibility
=================

The libc is written to be ABI independent.  Interfaces are generated using
LLVM's tablegen, so supporting arbitrary ABIs is possible.  In it's initial
stages there is no ABI stability in any form.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Using

   usage_modes
   overlay_mode
   fullbuild_mode

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Status

   date_and_time
   math
   strings
   stdio

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   build_system
   clang_tidy_checks
   entrypoints
   fuzzing
   ground_truth_specification
   header_generation
   implementation_standard
   api_test
   mechanics_of_public_api
   source_layout

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: External Links

   Source Code <https://github.com/llvm/llvm-project/tree/main/libc>
   Bug Reports <https://github.com/llvm/llvm-project/labels/libc>
   Buildbot <https://lab.llvm.org/buildbot/#/builders?tags=libc>
