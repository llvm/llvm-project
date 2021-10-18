=========================================
Libc++ 14.0.0 (In-Progress) Release Notes
=========================================

.. contents::
   :local:
   :depth: 2

Written by the `Libc++ Team <https://libcxx.llvm.org>`_

.. warning::

   These are in-progress notes for the upcoming libc++ 14 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the libc++ C++ Standard Library,
part of the LLVM Compiler Infrastructure, release 14.0.0. Here we describe the
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

What's New in Libc++ 14.0.0?
============================

New Features
------------

- There's initial support for the C++20 header ``<format>``. The implementation
  is incomplete. Some functions are known to be inefficient; both in memory
  usage and performance. The implementation is considered experimental and isn't
  considered ABI stable.

- There's a new CMake option ``LIBCXX_ENABLE_UNICODE`` to disable Unicode
  support in the ``<format>`` header. This only affects the estimation of the
  output width of the format functions.

- Support for building libc++ on top of a C Standard Library that does not support ``wchar_t`` was
  added. This is useful for building libc++ in an embedded setting, and it adds itself to the various
  freestanding-friendly options provided by libc++.

API Changes
-----------

- The functions ``std::atomic<T*>::fetch_(add|sub)`` and
  ``std::atomic_fetch_(add|sub)`` no longer accept a function pointer. While
  this is technically an API break, the invalid syntax isn't supported by
  libstc++ and MSVC STL.  See https://godbolt.org/z/49fvzz98d.

- The call of the functions ``std::atomic_(add|sub)(std::atomic<T*>*, ...)``
  with the explicit template argument ``T`` are now ill-formed. While this is
  technically an API break, the invalid syntax isn't supported by libstc++ and
  MSVC STL. See https://godbolt.org/z/v9959re3v.

  Due to this change it's now possible to call these functions with the
  explicit template argument ``T*``. This allows using the same syntax on the
  major Standard library implementations.
  See https://godbolt.org/z/oEfzPhTTb.

  Calls to these functions where the template argument was deduced by the
  compiler are unaffected by this change.

Build System Changes
--------------------

- Building the libc++ shared or static library requires a C++ 20 capable compiler.
  Consider using a Bootstrapping build to build libc++ with a fresh Clang if you
  can't use the system compiler to build libc++ anymore.

- Historically, there has been numerous ways of building libc++ and libc++abi. This has
  culminated in over 5 different ways to build the runtimes, which made it impossible to
  maintain with a good level of support. Starting with this release, the runtimes support
  exactly two ways of being built, which should cater to all use-cases. Furthermore,
  these builds are as lightweight as possible and will work consistently even when targetting
  embedded platforms, which used not to be the case. Please see the documentation on building
  libc++ to see those two ways of building and migrate over to the appropriate build instructions
  as soon as possible.

  All other ways to build are deprecated and will not be supported in the next release.
  We understand that making these changes can be daunting. For that reason, here's a
  summary of how to migrate from the two most common ways to build:

  - If you were rooting your CMake invocation at ``<monorepo>/llvm`` and passing ``-DLLVM_ENABLE_PROJECTS=<...>``
    (which was the previously advertised way to build the runtimes), please simply root your CMake invocation at
    ``<monorepo>/runtimes`` and pass ``-DLLVM_ENABLE_RUNTIMES=<...>``.

  - If you were doing two CMake invocations, one rooted at ``<monorepo>/libcxx`` and one rooted at
    ``<monorepo>/libcxxabi`` (this used to be called a "Standalone build"), please move them to a
    single invocation like so:

    .. code-block:: bash

        $ cmake -S <monorepo>/libcxx -B libcxx-build <LIBCXX-OPTIONS>
        $ cmake -S <monorepo>/libcxxabi -B libcxxabi-build <LIBCXXABI-OPTIONS>

    should become

    .. code-block:: bash

        $ cmake -S <monorepo>/runtimes -B build -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" <LIBCXX-OPTIONS> <LIBCXXABI-OPTIONS>
