=========================================
Libc++ 12.0.0 (In-Progress) Release Notes
=========================================

.. contents::
   :local:
   :depth: 2

Written by the `Libc++ Team <https://libcxx.llvm.org>`_

.. warning::

   These are in-progress notes for the upcoming libc++ 12 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the libc++ C++ Standard Library,
part of the LLVM Compiler Infrastructure, release 12.0.0. Here we describe the
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

What's New in Libc++ 12.0.0?
============================

New Features
------------
- Random device support has been made optional. It's enabled by default and can
  be disabled by building libc++ with ``-DLIBCXX_ENABLE_RANDOM_DEVICE=OFF``.
  Disabling random device support can be useful when building the library for
  platforms that don't have a source of randomness, such as some embedded
  platforms. When this is not supported, most of ``<random>`` will still be
  available, but ``std::random_device`` will not.
- Localization support has been made optional. It's enabled by default and can
  be disabled by building libc++ with ``-DLIBCXX_ENABLE_LOCALIZATION=OFF``.
  Disabling localization can be useful when porting to platforms that don't
  support the C locale API (e.g. embedded). When localization is not
  supported, several parts of the library will be disabled: ``<iostream>``,
  ``<regex>``, ``<locale>`` will be completely unusable, and other parts may be
  only partly available.
- If libc++ is compiled with a C++20 capable compiler it will be compiled in
  C++20 mode. Else libc++ will be compiled in C++17 mode.
- Several unqualified lookups in libc++ have been changed to qualified lookups.
  This makes libc++ more ADL-proof.
- The libc++ implementation status pages have been overhauled. Like other parts
  documentation they now use restructured text instead of html. Starting with
  libc++12 the status pages are part of libc++'s documentation.
- More C++20 features have been implemented. :doc:`Cxx2aStatus` has the full
  overview of libc++'s C++20 implementation status.
- Work has started to implement new C++2b features. :doc:`Cxx2bStatus` has the
  full overview of libc++'s C++2b implementation status.


API Changes
-----------
- By default, libc++ will _not_ include the definition for new and delete,
  since those are provided in libc++abi. Vendors wishing to provide new and
  delete in libc++ can build the library with ``-DLIBCXX_ENABLE_NEW_DELETE_DEFINITIONS=ON``
  to get back the old behavior. This was done to avoid providing new and delete
  in both libc++ and libc++abi, which is technically an ODR violation. Also
  note that we couldn't decide to put the operators in libc++ only, because
  they are needed from libc++abi (which would create a circular dependency).
- During the C++20 standardization process some new low-level bit functions
  have been renamed. Libc++ has renamed these functions to match the C++20
  Standard.
  - ``ispow2`` has been renamed to ``has_single_bit``
  - ``ceil2`` has been renamed to ``bit_ceil``
  - ``floor2`` has been renamed to ``bit_floor``
  - ``log2p1`` has been renamed to ``bit_width``

- In C++20 mode, ``std::filesystem::path::u8string()`` and
  ``generic_u8string()`` now return ``std::u8string`` according to P0428,
  while they return ``std::string`` in C++17. This can cause source
  incompatibility, which is discussed and acknowledged in P1423, but that
  paper doesn't suggest any remediation for this incompatibility.
