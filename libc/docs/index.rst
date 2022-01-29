=========================
"libc" C Standard Library
=========================

.. contents:: Table of Contents
  :depth: 4
  :local:

Goals
=====

llvm-libc will be developed to have a certain minimum set of features:

- C17 and upwards conformant.
- A modular libc with individual pieces implemented in the "as a
  library" philosophy of the LLVM project.
- Ability to layer this libc over the system libc if possible and desired
  for a platform.
- Provide C symbols as specified by the standards, but take advantage
  and use C++ language facilities for the core implementation.
- Provides POSIX extensions on POSIX compliant platforms.
- Provides system-specific extensions as appropriate. For example,
  provides the Linux API on Linux.
- Vendor extensions if and only if necessary.
- Designed and developed from the start to work with LLVM tooling and
  testing like fuzz testing and sanitizer-supported testing.
- ABI independent implementation as far as possible.
- Use source based implementations as far possible rather than
  assembly. Will try to *fix* the compiler rather than use assembly
  language workarounds.
- Extensive unit testing and standards conformance testing. If relevant
  and possible, differential testing: We want to be able
  to test llvm-libc against another battle-tested libc. This is
  essentially to understand how we differ from other libcs. Also if
  relevant and possible, test against the testsuite of an another
  battle-tested libc implementation.

Why a new C Standard Library?
=============================

Implementing a libc is no small task and is not be taken lightly. A
natural question to ask is, "why a new implementation of the C
standard library?" There is no single answer to this question, but
some of the major reasons are as follows:

- Most libc implementations are monolithic. It is a non-trivial
  porting task to pick and choose only the pieces relevant to one's
  platform. The llvm-libc will be developed with sufficient modularity to
  make picking and choosing a straightforward task.
- Most libc implementations break when built with sanitizer specific
  compiler options. The llvm-libc will be developed from the start to
  work with those specialized compiler options.
- The llvm-libc will be developed to support and employ fuzz testing
  from the start.
- Most libc implementations use a good amount of assembly language,
  and assume specific ABIs (may be platform dependent). With the llvm-libc
  implementation, we want to use normal source code as much as possible so
  that compiler-based changes to the ABI are easy. Moreover, as part of the
  LLVM project, we want to use this opportunity to fix performance related
  compiler bugs rather than using assembly workarounds.
- A large hole in the LLVM toolchain will be plugged with llvm-libc.
  With the broad platform expertise in the LLVM community, and the
  strong license and project structure, we think that llvm-libc will
  be more tunable and robust, without sacrificing the simplicity and
  accessibility typical of the LLVM project.

Platform Support
================

We envision that llvm-libc will support a variety of platforms in the coming
years. Interested parties are encouraged to participate in the design and
implementation, and add support for their favorite platforms.

ABI Compatibility
=================

As llvm-libc is new, it will not offer ABI stability in the initial stages.
However, as we've heard from other LLVM contributors that they are interested
in having ABI stability, llvm-libc code will be written in a manner which is
amenable to ABI stability. We are looking for contributors interested in
driving the design in this space to help us define what exactly does ABI
stability mean for llvm-libc.

Layering Over Another libc
==========================

When meaningful and practically possible on a platform, llvm-libc will be
developed in a fashion that it will be possible to layer it over the system
libc. This does not mean that one can mix llvm-libc with the system-libc. Also,
it does not mean that layering is the only way to use llvm-libc. What it
means is that, llvm-libc can optionally be packaged in a way that it can
delegate parts of the functionality to the system-libc. The delegation happens
internal to llvm-libc and is invisible to the users. From the user's point of
view, they only call into llvm-libc.

There are a few problems one needs to be mindful of when implementing such a
delegation scheme in llvm-libc. Examples of such problems are:

1. One cannot mix data structures from llvm-libc with those from the
system-libc. A translation from one set of data structures to the other should
happen internal to llvm-libc.
2. The delegation mechanism has to be implemented over a related set of
functions. For example, one cannot delegate just the `fopen` function to the
system-libc. One will have to delegate all `FILE` related functions to the
system-libc.

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
    mechanics_of_public_api
    redirectors
    source_layout
