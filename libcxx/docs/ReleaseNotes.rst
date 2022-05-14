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

- Implemented P0674R1 (Support arrays in ``make_shared`` and ``allocate_shared``)

- Implemented P0980R1 (Making ``std::string`` constexpr)

- Marked the following papers as "Complete" (note that some of those might have
  been implemented in a previous release but not marked as such):

    - P1207R4 (Movability of Single-pass Iterators);
    - P1474R1 (Helpful pointers for ``ContiguousIterator``);
    - P1522R1 (Iterator Difference Type and Integer Overflow);
    - P1523R1 (Views and Size Types);
    - P1456R1 (Move-only views);
    - P1870R1 (``forwarding-range`` is too subtle);
    - P1878R1 (Constraining Readable Types);
    - P1970R2 (Consistency for ``size()`` functions: Add ``ranges::ssize``);
    - P1983R0 (Wording for GB301, US296, US292, US291, and US283).

- `pop_heap` now uses an algorithm known as "bottom-up heapsort" or
  "heapsort with bounce" to reduce the number of comparisons, and rearranges
  elements using move-assignment instead of `swap`.

 - Libc++ now supports a variety of assertions that can be turned on to help catch
   undefined behavior in user code. This new support is now separate from the old
   (and incomplete) Debug Mode. Vendors can select whether the library they ship
   should include assertions or not by default. For details, see
   :ref:`the documentation <assertions-mode>` about this new feature.

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

- Some libc++ headers no longer transitively include all of:
    - ``<algorithm>``
    - ``<chrono>``
    - ``<functional>``
    - ``<utility>``

  If, after updating libc++, you see compiler errors related to missing declarations
  in namespace ``std``, it might be because one of your source files now needs to
  include one or more of the headers listed above.

- The integer distributions ``binomial_distribution``, ``discrete_distribution``,
  ``geometric_distribution``, ``negative_binomial_distribution``, ``poisson_distribution``,
  and ``uniform_int_distribution`` now conform to the Standard by rejecting
  template parameter types other than ``short``, ``int``, ``long``, ``long long``,
  (as an extension) ``__int128_t``, and the unsigned versions thereof.
  In particular, ``uniform_int_distribution<int8_t>`` is no longer supported.

- The C++14 function ``std::quoted(const char*)`` is no longer supported in
  C++03 or C++11 modes.

- Setting a custom debug handler with ``std::__libcpp_debug_function`` is not
  supported anymore. Please migrate to using the new support for
  :ref:`assertions <assertions-mode>` instead.

- ``vector<bool>::const_reference``, ``vector<bool>::const_iterator::reference``
  and ``bitset::const_reference`` are now aliases for `bool` in the unstable ABI.

ABI Changes
-----------

- The ``_LIBCPP_ABI_USE_CXX03_NULLPTR_EMULATION`` macro controlling whether we use an
  emulation for ``std::nullptr_t`` in C++03 mode has been removed. After this change,
  ``_LIBCPP_ABI_USE_CXX03_NULLPTR_EMULATION`` will not be honoured anymore and there
  will be no way to opt back into the C++03 emulation of ``std::nullptr_t``.

- On FreeBSD, NetBSD, DragonFlyBSD and Solaris, ``std::random_device`` is now implemented on
  top of ``arc4random()`` instead of reading from ``/dev/urandom``. Any implementation-defined
  token used when constructing a ``std::random_device`` will now be ignored instead of
  interpreted as a file to read entropy from.

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

- Previously, the C++ ABI library headers would be installed inside ``<prefix>/include/c++/v1``
  alongside the libc++ headers as part of building libc++. This is not the case anymore -- the
  ABI library is expected to install its headers where it wants them as part of its own build.
  Note that no action is required for most users, who build libc++ against libc++abi, since
  libc++abi already installs its headers in the right location. However, vendors building
  libc++ against alternate ABI libraries should make sure that their ABI library installs
  its own headers.

- The legacy testing configuration is now deprecated and will be removed in the next release. For
  most users, this should not have any impact. However, if you are testing libc++ in a configuration
  or on a platform that used to be supported by the legacy testing configuration and isn't supported
  by one of the configurations in ``libcxx/test/configs``, please reach out to the libc++ developers
  to get your configuration supported officially.
