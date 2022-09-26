=========================================
Libc++ 16.0.0 (In-Progress) Release Notes
=========================================

.. contents::
   :local:
   :depth: 2

Written by the `Libc++ Team <https://libcxx.llvm.org>`_

.. warning::

   These are in-progress notes for the upcoming libc++ 16 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the libc++ C++ Standard Library,
part of the LLVM Compiler Infrastructure, release 16.0.0. Here we describe the
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

What's New in Libc++ 16.0.0?
============================

Implemented Papers
------------------
- P2499R0 - ``string_view`` range constructor should be ``explicit``
- P2417R2 - A more constexpr bitset
- P2445R1 - ``std::forward_like``
- P2273R3 - Making ``std::unique_ptr`` constexpr

Improvements and New Features
-----------------------------
- Declarations of ``std::c8rtomb()`` and ``std::mbrtoc8()`` from P0482R6 are
  now provided when implementations in the global namespace are provided by
  the C library.

Deprecations and Removals
-------------------------
- ``unary_function`` and ``binary_function`` are no longer provided in C++17 and newer Standard modes.
  They can be re-enabled with ``_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION``.

- Several incidental transitive includes have been removed from libc++. Those
  includes are removed based on the language version used. Incidental transitive
  inclusions of the following headers have been removed:

  - C++20: ``chrono``
  - C++2b: ``algorithm``, ``array``, ``atomic``, ``bit``, ``chrono``,
    ``climits``, ``cmath``, ``compare``, ``concepts``, ``cstdarg`, ``cstddef``,
    ``cstdint``, ``cstdlib``, ``cstring``, ``ctime``, ``exception``,
    ``functional``, ``initializer_list``, ``iosfwd``, ``iterator``, ``limits``,
    ``memory``, ``new``, ``numeric``, ``optional``, ``ratio``, ``stdexcept``,
    ``string``, ``tuple``, ``type_traits``, ``typeinfo``, ``unordered_map``,
    ``utility``, ``variant``, ``vector``.

  Users can also remove all incidental transitive includes by defining
  ``_LIBCPP_REMOVE_TRANSITIVE_INCLUDES`` regardless of the language version
  in use. Note that in the future, libc++ reserves the right to remove
  incidental transitive includes more aggressively, in particular regardless
  of the language version in use.


Upcoming Deprecations and Removals
----------------------------------

API Changes
-----------
- The comparison operators on ``thread::id`` are now defined as free-standing
  functions instead of as hidden friends, in conformance with the C++ standard.
  Also see `issue 56187 <https://github.com/llvm/llvm-project/issues/56187>`_.

- ``_LIBCPP_ENABLE_NODISCARD`` and ``_LIBCPP_DISABLE_NODISCARD_AFTER_CXX17`` are no longer respected.
  Any standards-required ``[[nodiscard]]`` applications in C++20 are now always enabled. Any extended applications
  are now enabled by default and can be disabled by defining ``_LIBCPP_DISABLE_NODISCARD_EXT``.

ABI Affecting Changes
---------------------
- In freestanding mode, ``atomic<small enum class>`` does not contain a lock byte anymore if the platform
  can implement lockfree atomics for that size. More specifically, in LLVM <= 11.0.1, an ``atomic<small enum class>``
  would not contain a lock byte. This was broken in LLVM >= 12.0.0, where it started including a lock byte despite
  the platform supporting lockfree atomics for that size. Starting in LLVM 15.0.1, the ABI for these types has been
  restored to what it used to be (no lock byte), which is the most efficient implementation.

  This ABI break only affects users that compile with ``-ffreestanding``, and only for ``atomic<T>`` where ``T``
  is a non-builtin type that could be lockfree on the platform. See https://llvm.org/D133377 for more details.

Build System Changes
--------------------
