==================
The LLVM C Library
==================

.. warning::
  LLVM-libc is not yet ABI stable; currently only static linking is supported.
  LLVM-libc developers retain the right to modify the ABI of types used
  throughout the library. Another libc should be preferred if ABI stability is
  a requirement.

.. note::
  LLVM-libc is not fully complete right now. Some programs may fail to build due
  to missing functions. If you would like to help us finish LLVM-libc, check
  out "`Contributing to the libc project <contributing.html>`__" in the sidebar
  or ask on `discord <https://discord.com/channels/636084430946959380/636732994891284500>`__.

Introduction
============

LLVM-libc aspires to a unique place in the software ecosystem.  The goals are:

- Fully compliant with current C23 and POSIX.1-2024 standards.
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
- `Fuzzing <https://github.com/llvm/llvm-project/tree/main/libc/fuzzing>`__


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Status & Support

   headers/index.rst
   arch_support
   platform_support
   compiler_support

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Simple Usage

   getting_started

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Advanced Usage

   full_host_build
   full_cross_build
   overlay_mode
   gpu/index.rst
   configure

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   build_and_test
   dev/index.rst
   porting
   contributing

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Useful Links

   talks
   Source Code <https://github.com/llvm/llvm-project/tree/main/libc>
   Bug Reports <https://github.com/llvm/llvm-project/labels/libc>
   Discourse <https://discourse.llvm.org/c/runtimes/libc>
   Join the Discord <https://discord.gg/xS7Z362>
   Discord Channel <https://discord.com/channels/636084430946959380/636732994891284500>
   Buildbot <https://lab.llvm.org/buildbot/#/builders?tags=libc>
