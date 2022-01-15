====================================================
Extra Clang Tools 14.0.0 (In-Progress) Release Notes
====================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 14 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 14.0.0. Here we describe the status of the Extra Clang Tools in
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

What's New in Extra Clang Tools 14.0.0?
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

- Make the `cppcoreguidelines-pro-bounds-array-to-pointer-decay` check accept
  string literal to pointer decay in conditional operator even if operands are
  of the same length.

- Ignore warnings from macros defined in system headers, if not using the
  `-system-headers` flag.

- Added support for globbing in `NOLINT*` expressions, to simplify suppressing
  multiple warnings in the same line.

- Added support for `NOLINTBEGIN` ... `NOLINTEND` comments to suppress
  Clang-Tidy warnings over multiple lines.

- Generalized the `modernize-use-default-member-init` check to handle non-default
  constructors.

- Eliminated false positives for `cppcoreguidelines-macro-usage` by restricting
  the warning about using constants to only macros that expand to literals.

New checks
^^^^^^^^^^

- New :doc:`bugprone-stringview-nullptr
  <clang-tidy/checks/bugprone-stringview-nullptr>` check.

  Checks for various ways that the ``const CharT*`` constructor of
  ``std::basic_string_view`` can be passed a null argument.

- New :doc:`abseil-cleanup-ctad
  <clang-tidy/checks/abseil-cleanup-ctad>` check.

  Suggests switching the initialization pattern of ``absl::Cleanup``
  instances from the factory function to class template argument
  deduction (CTAD), in C++17 and higher.

- New :doc:`bugprone-suspicious-memory-comparison
  <clang-tidy/checks/bugprone-suspicious-memory-comparison>` check.

  Finds potentially incorrect calls to ``memcmp()`` based on properties of the
  arguments.

- New :doc:`cppcoreguidelines-virtual-class-destructor
  <clang-tidy/checks/cppcoreguidelines-virtual-class-destructor>` check.

  Finds virtual classes whose destructor is neither public and virtual nor
  protected and non-virtual.

- New :doc:`misc-misleading-identifier <clang-tidy/checks/misc-misleading-identifier>` check.

  Reports identifier with unicode right-to-left characters.

- New :doc:`readability-container-contains
  <clang-tidy/checks/readability-container-contains>` check.

  Finds usages of ``container.count()`` and ``container.find() == container.end()`` which should
  be replaced by a call to the ``container.contains()`` method introduced in C++20.

- New :doc:`readability-container-data-pointer
  <clang-tidy/checks/readability-container-data-pointer>` check.

  Finds cases where code could use ``data()`` rather than the address of the
  element at index 0 in a container.

- New :doc:`readability-duplicate-include
  <clang-tidy/checks/readability-duplicate-include>` check.

  Looks for duplicate includes and removes them.

- New :doc:`readability-identifier-length
  <clang-tidy/checks/readability-identifier-length>` check.

  Reports identifiers whose names are too short. Currently checks local
  variables and function parameters only.

- New :doc:`misc-misleading-bidirectional <clang-tidy/checks/misc-misleading-bidirectional>` check.

  Inspects string literal and comments for unterminated bidirectional Unicode
  characters.

New check aliases
^^^^^^^^^^^^^^^^^

- New alias :doc:`cert-err33-c
  <clang-tidy/checks/cert-err33-c>` to
  :doc:`bugprone-unused-return-value
  <clang-tidy/checks/bugprone-unused-return-value>` was added.

- New alias :doc:`cert-exp42-c
  <clang-tidy/checks/cert-exp42-c>` to
  :doc:`bugprone-suspicious-memory-comparison
  <clang-tidy/checks/bugprone-suspicious-memory-comparison>` was added.

- New alias :doc:`cert-flp37-c
  <clang-tidy/checks/cert-flp37-c>` to
  :doc:`bugprone-suspicious-memory-comparison
  <clang-tidy/checks/bugprone-suspicious-memory-comparison>` was added.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- :doc:`bugprone-assert-side-effect <clang-tidy/checks/bugprone-assert-side-effect>`
  check now supports an ``IgnoredFunctions`` option to explicitly consider
  the specified semicolon-separated functions list as not having any
  side-effects. Regular expressions for the list items are also accepted.

- Removed default setting ``cppcoreguidelines-explicit-virtual-functions.IgnoreDestructors = "true"``,
  from :doc:`cppcoreguidelines-explicit-virtual-functions <clang-tidy/checks/cppcoreguidelines-explicit-virtual-functions>`
  to match the current state of the C++ Core Guidelines.

- Removed suggestion ``use gsl::at`` from warning message in the
  ``cppcoreguidelines-pro-bounds-constant-array-index`` check, since that is not
  a requirement from the C++ Core Guidelines. This allows people to choose
  their own safe indexing strategy. The fix-it is kept for those who want to
  use the GSL library.

- Updated :doc:`google-readability-casting
  <clang-tidy/checks/google-readability-casting>` to diagnose and fix functional
  casts, to achieve feature parity with the corresponding ``cpplint.py`` check.

- Fixed a false positive in :doc:`fuchsia-trailing-return
  <clang-tidy/checks/fuchsia-trailing-return>` for C++17 deduction guides.

- Fixed a false positive in :doc:`bugprone-throw-keyword-missing
  <clang-tidy/checks/bugprone-throw-keyword-missing>` when creating an exception object
  using placement new.

- :doc:`cppcoreguidelines-narrowing-conversions <clang-tidy/checks/cppcoreguidelines-narrowing-conversions>`
  check now supports a ``WarnOnIntegerToFloatingPointNarrowingConversion``
  option to control whether to warn on narrowing integer to floating-point
  conversions.

- Improved :doc:`performance-move-const-arg <clang-tidy/checks/performance-move-const-arg>` check.

  Removed a wrong FixIt for trivially copyable objects wrapped by ``std::move()`` and passed to an rvalue reference parameter. Removal of ``std::move()`` would break the code.

Removed checks
^^^^^^^^^^^^^^

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

Clang-tidy Visual Studio plugin
-------------------------------
