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

...

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

Hover
^^^^^

Code completion
^^^^^^^^^^^^^^^

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

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- Preprocessor-level module header parsing is now disabled by default due to
  the problems it caused in C++20 and above, leading to performance and code
  parsing issues regardless of whether modules were used or not. This change
  will impact only the following checks:
  :doc:`modernize-replace-disallow-copy-and-assign-macro
  <clang-tidy/checks/modernize/replace-disallow-copy-and-assign-macro>`,
  :doc:`bugprone-reserved-identifier
  <clang-tidy/checks/bugprone/reserved-identifier>`, and
  :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>`. Those checks will no
  longer see macros defined in modules. Users can still enable this
  functionality using the newly added command line option
  `--enable-module-headers-parsing`.

- Remove configuration option `AnalyzeTemporaryDestructors`, which was deprecated since
  :program:`clang-tidy` 16.

- Improved `--dump-config` to print check options in alphabetical order.

New checks
^^^^^^^^^^

- New :doc:`bugprone-inc-dec-in-conditions
  <clang-tidy/checks/bugprone/inc-dec-in-conditions>` check.

  Detects when a variable is both incremented/decremented and referenced inside
  a complex condition and suggests moving them outside to avoid ambiguity in
  the variable's value.

- New :doc:`bugprone-multi-level-implicit-pointer-conversion
  <clang-tidy/checks/bugprone/multi-level-implicit-pointer-conversion>` check.

  Detects implicit conversions between pointers of different levels of
  indirection.

- New :doc:`bugprone-optional-value-conversion
  <clang-tidy/checks/bugprone/optional-value-conversion>` check.

  Detects potentially unintentional and redundant conversions where a value is
  extracted from an optional-like type and then used to create a new instance
  of the same optional-like type.

- New :doc:`modernize-use-constraints
  <clang-tidy/checks/modernize/use-constraints>` check.

  Replace ``enable_if`` with C++20 requires clauses.

- New :doc:`performance-enum-size
  <clang-tidy/checks/performance/enum-size>` check.

  Recommends the smallest possible underlying type for an ``enum`` or ``enum``
  class based on the range of its enumerators.

- New :doc:`readability-reference-to-constructed-temporary
  <clang-tidy/checks/readability/reference-to-constructed-temporary>` check.

  Detects C++ code where a reference variable is used to extend the lifetime
  of a temporary object that has just been constructed.

New check aliases
^^^^^^^^^^^^^^^^^

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed bug in :doc:`bugprone-reserved-identifier
  <clang-tidy/checks/bugprone/reserved-identifier>`, so that it does not warn
  on macros starting with underscore and lowercase letter.

- Improved :doc:`llvm-namespace-comment
  <clang-tidy/checks/llvm/namespace-comment>` check to provide fixes for
  ``inline`` namespaces in the same format as :program:`clang-format`.

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

Clang-tidy Visual Studio plugin
-------------------------------
