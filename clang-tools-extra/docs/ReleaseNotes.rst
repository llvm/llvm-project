====================================================
Extra Clang Tools |release| |ReleaseNotesTitle|
====================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming Extra Clang Tools |version| release.
     Release notes for previous releases can be found on
     `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release |release|. Here we describe the status of the Extra Clang Tools in
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

What's New in Extra Clang Tools |release|?
==========================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

Improvements to clangd
----------------------

Inlay hints
^^^^^^^^^^^

Diagnostics
^^^^^^^^^^^

Semantic Highlighting
^^^^^^^^^^^^^^^^^^^^^

Compile flags
^^^^^^^^^^^^^

- Added `BuiltinHeaders` config key which controls whether clangd's built-in
  headers are used or ones extracted from the driver.

Hover
^^^^^

Code completion
^^^^^^^^^^^^^^^

Code actions
^^^^^^^^^^^^

Signature help
^^^^^^^^^^^^^^

Cross-references
^^^^^^^^^^^^^^^^

Objective-C
^^^^^^^^^^^

Miscellaneous
^^^^^^^^^^^^^

Improvements to clang-doc
-------------------------

Improvements to clang-query
---------------------------

Improvements to include-cleaner
-------------------------------
- Deprecated the ``-insert`` and ``-remove`` command line options, and added
  the ``-disable-remove`` and ``-disable-insert`` command line options as
  replacements. The previous command line options were confusing because they
  did not imply the default state of the option (which is inserts and removes
  being enabled). The new options are easier to understand the semantics of.

Improvements to clang-tidy
--------------------------

- Changed the :program:`check_clang_tidy.py` tool to use FileCheck's
  ``--match-full-lines`` instead of ``strict-whitespace`` for ``CHECK-FIXES``
  clauses. Added a ``--match-partial-fixes`` option to keep previous behavior on
  specific tests. This may break tests for users with custom out-of-tree checks
  who use :program:`check_clang_tidy.py` as-is.

- Improved :program:`clang-tidy-diff.py` script. Add the `-warnings-as-errors`
  argument to treat warnings as errors.

- Fixed bug in :program:`clang-tidy` by which `HeaderFilterRegex` did not take
  effect when passed via the `.clang-tidy` file.

- Fixed bug in :program:`run_clang_tidy.py` where the program would not
  correctly display the checks enabled by the top-level `.clang-tidy` file.

New checks
^^^^^^^^^^

- New :doc:`bugprone-capturing-this-in-member-variable
  <clang-tidy/checks/bugprone/capturing-this-in-member-variable>` check.

  Finds lambda captures and ``bind`` function calls that capture the ``this``
  pointer and store it as class members without handle the copy and move
  constructors and the assignments.

- New :doc:`bugprone-misleading-setter-of-reference
  <clang-tidy/checks/bugprone/misleading-setter-of-reference>` check.

  Finds setter-like member functions that take a pointer parameter and set a
  reference member of the same class with the pointed value.

- New :doc:`bugprone-unintended-char-ostream-output
  <clang-tidy/checks/bugprone/unintended-char-ostream-output>` check.

  Finds unintended character output from ``unsigned char`` and ``signed char``
  to an ``ostream``.

- New :doc:`readability-ambiguous-smartptr-reset-call
  <clang-tidy/checks/readability/ambiguous-smartptr-reset-call>` check.

  Finds potentially erroneous calls to ``reset`` method on smart pointers when
  the pointee type also has a ``reset`` method.

New check aliases
^^^^^^^^^^^^^^^^^

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-crtp-constructor-accessibility
  <clang-tidy/checks/bugprone/crtp-constructor-accessibility>` check by fixing
  false positives on deleted constructors that cannot be used to construct
  objects, even if they have public or protected access.

- Improved :doc:`bugprone-optional-value-conversion
  <clang-tidy/checks/bugprone/optional-value-conversion>` check to detect
  conversion in argument of ``std::make_optional``.

- Improved :doc:`bugprone-string-constructor
  <clang-tidy/checks/bugprone/string-constructor>` check to find suspicious
  calls of ``std::string`` constructor with char pointer, start position and
  length parameters.

- Improved :doc:`bugprone-unchecked-optional-access
  <clang-tidy/checks/bugprone/unchecked-optional-access>` fixing false
  positives from smart pointer accessors repeated in checking ``has_value``
  and accessing ``value``. The option `IgnoreSmartPointerDereference` should
  no longer be needed and will be removed. Also fixing false positive from
  const reference accessors to objects containing optional member.

- Improved :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` check to allow specifying
  additional C++ member functions to match.

- Improved :doc:`cert-err33-c
  <clang-tidy/checks/cert/err33-c>` check by fixing false positives when
  a function name is just prefixed with a targeted function name.

- Improved :doc:`concurrency-mt-unsafe
  <clang-tidy/checks/concurrency/mt-unsafe>` check by fixing a false positive
  where ``strerror`` was flagged as MT-unsafe.

- Improved :doc:`misc-const-correctness
  <clang-tidy/checks/misc/const-correctness>` check by adding the option
  `AllowedTypes`, that excludes specified types from const-correctness
  checking and fixing false positives when modifying variant by ``operator[]``
  with template in parameters and supporting to check pointee mutation by
  `AnalyzePointers` option and fixing false positives when using const array
  type.

- Improved :doc:`misc-include-cleaner
  <clang-tidy/checks/misc/include-cleaner>` check by adding the options
  `UnusedIncludes` and `MissingIncludes`, which specify whether the check should
  report unused or missing includes respectively.

- Improved :doc:`misc-redundant-expression
  <clang-tidy/checks/misc/redundant-expression>` check by providing additional
  examples and fixing some macro related false positives.

- Improved :doc:`misc-unused-using-decls
  <clang-tidy/checks/misc/unused-using-decls>` check by fixing false positives
  on ``operator""`` with template parameters.

- Improved :doc:`misc-use-internal-linkage
  <clang-tidy/checks/misc/use-internal-linkage>` check by fix false positives
  for function or variable in header file which contains macro expansion and
  excluding variables with ``thread_local`` storage class specifier from being
  matched.

- Improved :doc:`modernize-use-default-member-init
  <clang-tidy/checks/modernize/use-default-member-init>` check by matching
  arithmetic operations, ``constexpr`` and ``static`` values, and detecting
  explicit casting of built-in types within member list initialization.

- Improved :doc:`modernize-use-designated-initializers
  <clang-tidy/checks/modernize/use-designated-initializers>` check by avoiding
  diagnosing designated initializers for ``std::array`` initializations.

- Improved :doc:`modernize-use-ranges
  <clang-tidy/checks/modernize/use-ranges>` check by updating suppress
  warnings logic for ``nullptr`` in ``std::find``.

- Improved :doc:`modernize-use-starts-ends-with
  <clang-tidy/checks/modernize/use-starts-ends-with>` check by adding more
  matched scenarios of ``find`` and ``rfind`` methods and fixing false
  positives when those methods were called with 3 arguments.

- Improved :doc:`modernize-use-std-numbers
  <clang-tidy/checks/modernize/use-std-numbers>` check to support math
  functions of different precisions.

- Improved :doc:`performance-move-const-arg
  <clang-tidy/checks/performance/move-const-arg>` check by fixing false
  negatives on ternary operators calling ``std::move``.

- Improved :doc:`performance-unnecessary-value-param
  <clang-tidy/checks/performance/unnecessary-value-param>` check performance by
  tolerating fix-it breaking compilation when functions is used as pointers
  to avoid matching usage of functions within the current compilation unit.

- Improved :doc:`readability-qualified-auto
  <clang-tidy/checks/readability/qualified-auto>` check by adding the option
  `AllowedTypes`, that excludes specified types from adding qualifiers.

Removed checks
^^^^^^^^^^^^^^

Miscellaneous
^^^^^^^^^^^^^

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

Clang-tidy Visual Studio plugin
-------------------------------
