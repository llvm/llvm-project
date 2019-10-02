====================================================
Extra Clang Tools 11.0.0 (In-Progress) Release Notes
====================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 11 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 11.0.0. Here we describe the status of the Extra Clang Tools in
some detail, including major improvements from the previous release and new
feature work. All LLVM releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or
the `LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Extra Clang Tools 11.0.0?
=======================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

...

Improvements to clangd
----------------------

The improvements are...

Improvements to clang-doc
-------------------------

The improvements are...

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

New module
^^^^^^^^^^
- New module `llvmlibc`.

  This module contains checks related to the LLVM-libc coding standards.

New checks
^^^^^^^^^^

- New :doc:`abseil-string-find-str-contains
  <clang-tidy/checks/abseil-string-find-str-contains>` check.

  Finds ``s.find(...) == string::npos`` comparisons (for various string-like types)
  and suggests replacing with ``absl::StrContains()``.

- New :doc:`cppcoreguidelines-avoid-non-const-global-variables
  <clang-tidy/checks/cppcoreguidelines-avoid-non-const-global-variables>` check.
  Finds non-const global variables as described in check I.2 of C++ Core
  Guidelines.

- New :doc:`bugprone-misplaced-pointer-arithmetic-in-alloc
  <clang-tidy/checks/bugprone-misplaced-pointer-arithmetic-in-alloc>` check.

  Finds cases where an integer expression is added to or subtracted from the
  result of a memory allocation function (``malloc()``, ``calloc()``,
  ``realloc()``, ``alloca()``) instead of its argument.

- New :doc:`bugprone-spuriously-wake-up-functions
  <clang-tidy/checks/bugprone-spuriously-wake-up-functions>` check.

  Finds ``cnd_wait``, ``cnd_timedwait``, ``wait``, ``wait_for``, or
  ``wait_until`` function calls when the function is not invoked from a loop
  that checks whether a condition predicate holds or the function has a
  condition parameter.

- New :doc:`bugprone-reserved-identifier
  <clang-tidy/checks/bugprone-reserved-identifier>` check.

  Checks for usages of identifiers reserved for use by the implementation.

- New :doc:`bugprone-suspicious-include
  <clang-tidy/checks/bugprone-suspicious-include>` check.

  Finds cases where an include refers to what appears to be an implementation
  file, which often leads to hard-to-track-down ODR violations, and diagnoses
  them.

- New :doc:`cert-oop57-cpp
  <clang-tidy/checks/cert-oop57-cpp>` check.

  Flags use of the `C` standard library functions ``memset``, ``memcpy`` and
  ``memcmp`` and similar derivatives on non-trivial types.

- New :doc:`llvmlibc-callee-namespace
  <clang-tidy/checks/llvmlibc-callee-namespace>` check.

  Checks all calls resolve to functions within ``__llvm_libc`` namespace.

- New :doc:`llvmlibc-implementation-in-namespace
  <clang-tidy/checks/llvmlibc-implementation-in-namespace>` check.

  Checks all llvm-libc implementation is within the correct namespace.

- New :doc:`llvmlibc-restrict-system-libc-headers
  <clang-tidy/checks/llvmlibc-restrict-system-libc-headers>` check.

  Finds includes of system libc headers not provided by the compiler within
  llvm-libc implementations.

- New :doc:`objc-dealloc-in-category
  <clang-tidy/checks/objc-dealloc-in-category>` check.

  Finds implementations of -dealloc in Objective-C categories.

- New :doc:`misc-no-recursion
  <clang-tidy/checks/misc-no-recursion>` check.

  Finds recursive functions and diagnoses them.

- New :doc:`objc-nsinvocation-argument-lifetime
  <clang-tidy/checks/objc-nsinvocation-argument-lifetime>` check.

  Finds calls to ``NSInvocation`` methods under ARC that don't have proper
  argument object lifetimes.

- New :doc:`readability-use-anyofallof
  <clang-tidy/checks/readability-use-anyofallof>` check.

  Finds range-based for loops that can be replaced by a call to ``std::any_of``
  or ``std::all_of``.

New check aliases
^^^^^^^^^^^^^^^^^

- New alias :doc:`cert-con36-c
  <clang-tidy/checks/cert-con36-c>` to
  :doc:`bugprone-spuriously-wake-up-functions
  <clang-tidy/checks/bugprone-spuriously-wake-up-functions>` was added.

- New alias :doc:`cert-con54-cpp
  <clang-tidy/checks/cert-con54-cpp>` to
  :doc:`bugprone-spuriously-wake-up-functions
  <clang-tidy/checks/bugprone-spuriously-wake-up-functions>` was added.

- New alias :doc:`cert-dcl37-c
  <clang-tidy/checks/cert-dcl37-c>` to
  :doc:`bugprone-reserved-identifier
  <clang-tidy/checks/bugprone-reserved-identifier>` was added.

- New alias :doc:`cert-dcl51-cpp
  <clang-tidy/checks/cert-dcl51-cpp>` to
  :doc:`bugprone-reserved-identifier
  <clang-tidy/checks/bugprone-reserved-identifier>` was added.

- New alias :doc:`cert-str34-c
  <clang-tidy/checks/cert-str34-c>` to
  :doc:`bugprone-signed-char-misuse
  <clang-tidy/checks/bugprone-signed-char-misuse>` was added.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:'readability-identifier-naming
  <clang-tidy/checks/readability-identifier-naming>` check.

  Now able to rename member references in class template definitions with 
  explicit access.

- Improved :doc:`readability-qualified-auto
  <clang-tidy/checks/readability-qualified-auto>` check now supports a
  `AddConstToQualified` to enable adding ``const`` qualifiers to variables
  typed with ``auto *`` and ``auto &``.

- Improved :doc:`readability-redundant-string-init
  <clang-tidy/checks/readability-redundant-string-init>` check now supports a
  `StringNames` option enabling its application to custom string classes. The
  check now detects in class initializers and constructor initializers which
  are deemed to be redundant.

- Checks supporting the ``HeaderFileExtensions`` flag now support ``;`` as a
  delimiter in addition to ``,``, with the latter being deprecated as of this
  release. This simplifies how one specifies the options on the command line:
  ``--config="{CheckOptions: [{ key: HeaderFileExtensions, value: h;;hpp;hxx }]}"``

Renamed checks
^^^^^^^^^^^^^^

- The 'fuchsia-restrict-system-headers' check was renamed to :doc:`portability-restrict-system-includes
  <clang-tidy/checks/portability-restrict-system-includes>`

Other improvements
^^^^^^^^^^^^^^^^^^

- For 'run-clang-tidy.py' add option to use alpha checkers from clang-analyzer.

Improvements to include-fixer
-----------------------------

The improvements are...

Improvements to clang-include-fixer
-----------------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...

Improvements to pp-trace
------------------------

The improvements are...

Clang-tidy visual studio plugin
-------------------------------
