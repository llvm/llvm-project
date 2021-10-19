=========================================
Libc++ 13.0.0 Release Notes
=========================================

.. contents::
   :local:
   :depth: 2

Written by the `Libc++ Team <https://libcxx.llvm.org>`_

Introduction
============

This document contains the release notes for the libc++ C++ Standard Library,
part of the LLVM Compiler Infrastructure, release 13.0.0. Here we describe the
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

What's New in Libc++ 13.0.0?
============================

- Support for older compilers has been removed. Several additional platforms
  are now officially supported. :ref:`platform_and_compiler_support` contains
  the complete overview of platforms and compilers supported by libc++.
- The large headers ``<algorithm>``, ``<iterator>``, and ``<utility>`` have
  been split in more granular headers. This reduces the size of included code
  when using libc++. This may lead to missing includes after upgrading to
  libc++13.

New Features
------------

- ``std::filesystem`` is now feature complete for the Windows platform using
  MinGW. MSVC isn't supported since it lacks 128-bit integer support.
- The implementation of the C++20 concepts library has been completed.
- Several C++20 ``constexpr`` papers have been completed:

  - `P0879R0 <https://wg21.link/P0879R0>`_ ``constexpr`` for ``std::swap()``
    and swap related functions
  - `P1032R1 <https://wg21.link/P1032R1>`_ Misc ``constexpr`` bits
  - `P0883 <https://wg21.link/P0883>`_ Fixing Atomic Initialization

- More C++20 features have been implemented. :doc:`Status/Cxx20` has the full
  overview of libc++'s C++20 implementation status.
- More C++2b features have been implemented. :doc:`Status/Cxx2b` has the
  full overview of libc++'s C++2b implementation status.
- The CMake option ``LIBCXX_ENABLE_INCOMPLETE_FEATURES`` has been added. This
  option allows libc++ vendors to disable headers that aren't production
  quality yet. Currently, turning the option off disables the headers
  ``<format>`` and ``<ranges>``.
- The documentation conversion from html to restructured text has been
  completed.

API Changes
-----------

- There has been several changes in the tuple constructors provided by libc++.
  Those changes were made as part of an effort to regularize libc++'s tuple
  implementation, which contained several subtle bugs due to these extensions.
  If you notice a build breakage when initializing a tuple, make sure you
  properly initialize all the tuple elements - this is probably the culprit.

  In particular, the extension allowing tuples to be constructed from fewer
  elements than the number of elements in the tuple (in which case the remaining
  elements would be default-constructed) has been removed. See https://godbolt.org/z/sqozjd.

  Also, the extension allowing a tuple to be constructed from an array has been
  removed. See https://godbolt.org/z/5esqbW.

- The ``std::pointer_safety`` utility and related functions are not available
  in C++03 anymore. Furthermore, in other standard modes, it has changed from
  a struct to a scoped enumeration, which is an ABI break. Finally, the
  ``std::get_pointer_safety`` function was previously in the dylib, but it
  is now defined as inline in the headers.

  While this is technically both an API and an ABI break, we do not expect
  ``std::pointer_safety`` to have been used at all in real code, since we
  never implemented the underlying support for garbage collection.

- The `LIBCXXABI_ENABLE_PIC` CMake option was removed. If you are building your
  own libc++abi from source and were using `LIBCXXABI_ENABLE_PIC`, please use
  `CMAKE_POSITION_INDEPENDENT_CODE=ON` instead.

- When the header <variant> is included, it will no longer include <array> transitively.

- The ``std::result_of`` and ``std::is_literal_type`` type traits have been removed in
  C++20 mode.
