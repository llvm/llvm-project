=========================================
Libc++ 15.0.0 (In-Progress) Release Notes
=========================================

.. contents::
   :local:
   :depth: 2

Written by the `Libc++ Team <https://libcxx.llvm.org>`_

.. warning::

   These are in-progress notes for the upcoming libc++ 15 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the libc++ C++ Standard Library,
part of the LLVM Compiler Infrastructure, release 15.0.0. Here we describe the
status of libc++ in some detail, including major improvements from the previous
release and new feature work. For the general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM releases may
be downloaded from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about libc++, please see the `Libc++ Web Site
<https://libcxx.llvm.org>`_ or the `LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Libc++ web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Libc++ 15.0.0?
============================

New Features
------------

 - Implemented P0627R6 (Function to mark unreachable code)

 - Implemented P1165R1 (Make stateful allocator propagation more consistent for ``operator+(basic_string)``)

 - `pop_heap` now uses an algorithm known as "bottom-up heapsort" or
   "heapsort with bounce" to reduce the number of comparisons, and rearranges
   elements using move-assignment instead of `swap`.

API Changes
-----------

- The ``_LIBCPP_ABI_UNSTABLE`` macro has been removed in favour of setting
  ``_LIBCPP_ABI_VERSION=2``. This should not have any impact on users because
  they were not supposed to set ``_LIBCPP_ABI_UNSTABLE`` manually, however we
  still feel that it is worth mentioning in the release notes in case some users
  had been doing it.

- The header ``<experimental/filesystem>`` has been removed. Instead, use
  ``<filesystem>`` header. The associated macro
  ``_LIBCPP_DEPRECATED_EXPERIMENTAL_FILESYSTEM`` has also been removed.

- Some libc++ headers no longer transitively include all of ``<algorithm>``and ``<chrono>``.
  If, after updating libc++, you see compiler errors related to missing declarations in
  namespace ``std``, it might be because one of your source files now needs to
  ``#include <algorithm>`` and/or ``#include <chrono>``.

- The integer distributions ``binomial_distribution``, ``discrete_distribution``,
  ``geometric_distribution``, ``negative_binomial_distribution``, ``poisson_distribution``,
  and ``uniform_int_distribution`` now conform to the Standard by rejecting
  template parameter types other than ``short``, ``int``, ``long``, ``long long``,
  (as an extension) ``__int128_t``, and the unsigned versions thereof.
  In particular, ``uniform_int_distribution<int8_t>`` is no longer supported.

- The C++14 function ``std::quoted(const char*)`` is no longer supported in
  C++03 or C++11 modes.

ABI Changes
-----------

- The ``_LIBCPP_ABI_USE_CXX03_NULLPTR_EMULATION`` macro controlling whether we use an
  emulation for ``std::nullptr_t`` in C++03 mode has been removed. After this change,
  ``_LIBCPP_ABI_USE_CXX03_NULLPTR_EMULATION`` will not be honoured anymore and there
  will be no way to opt back into the C++03 emulation of ``std::nullptr_t``.

Build System Changes
--------------------

- Support for standalone builds have been entirely removed from libc++, libc++abi and
  libunwind. Please use :ref:`these instructions <build instructions>` for building
  libc++, libc++abi and/or libunwind.

- The ``{LIBCXX,LIBCXXABI,LIBUNWIND}_TARGET_TRIPLE``, ``{LIBCXX,LIBCXXABI,LIBUNWIND}_SYSROOT`` and
  ``{LIBCXX,LIBCXXABI,LIBUNWIND}_GCC_TOOLCHAIN`` CMake variables have been removed. Instead, please
  use the ``CMAKE_CXX_COMPILER_TARGET``, ``CMAKE_SYSROOT`` and ``CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN``
  variables provided by CMake.

- When building for Windows, vendors who want to avoid dll-exporting symbols from the static libc++abi
  library should set ``LIBCXXABI_HERMETIC_STATIC_LIBRARY=ON`` when configuring CMake. The current
  behavior, which tries to guess the correct dll-export semantics based on whether we're building
  the libc++ shared library, will be removed in LLVM 16.
