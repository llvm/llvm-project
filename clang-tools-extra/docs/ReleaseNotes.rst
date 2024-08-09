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

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

New checks
^^^^^^^^^^


- New :doc:`boost-use-ranges
  <clang-tidy/checks/boost/use-ranges>` check.

  Detects calls to standard library iterator algorithms that could be replaced
  with a Boost ranges version instead.

- New :doc:`bugprone-crtp-constructor-accessibility
  <clang-tidy/checks/bugprone/crtp-constructor-accessibility>` check.

  Detects error-prone Curiously Recurring Template Pattern usage, when the CRTP
  can be constructed outside itself and the derived class.

- New :doc:`bugprone-pointer-arithmetic-on-polymorphic-object
  <clang-tidy/checks/bugprone/pointer-arithmetic-on-polymorphic-object>` check.

  Finds pointer arithmetic performed on classes that contain a virtual function.

- New :doc:`bugprone-return-const-ref-from-parameter
  <clang-tidy/checks/bugprone/return-const-ref-from-parameter>` check.

  Detects return statements that return a constant reference parameter as constant
  reference. This may cause use-after-free errors if the caller uses xvalues as
  arguments.

- New :doc:`bugprone-suspicious-stringview-data-usage
  <clang-tidy/checks/bugprone/suspicious-stringview-data-usage>` check.

  Identifies suspicious usages of ``std::string_view::data()`` that could lead
  to reading out-of-bounds data due to inadequate or incorrect string null
  termination.

- New :doc:`misc-use-internal-linkage
  <clang-tidy/checks/misc/use-internal-linkage>` check.

  Detects variables and functions that can be marked as static or moved into
  an anonymous namespace to enforce internal linkage.

- New :doc:`modernize-min-max-use-initializer-list
  <clang-tidy/checks/modernize/min-max-use-initializer-list>` check.

  Replaces nested ``std::min`` and ``std::max`` calls with an initializer list
  where applicable.

- New :doc:`modernize-use-cpp-style-comments
  <clang-tidy/checks/modernize/use-cpp-style-comments>` check.

  Detects C Style comments and suggests to use C++ style comments instead.

- New :doc:`modernize-use-designated-initializers
  <clang-tidy/checks/modernize/use-designated-initializers>` check.

  Finds initializer lists for aggregate types that could be
  written as designated initializers instead.

- New :doc:`modernize-use-ranges
  <clang-tidy/checks/modernize/use-ranges>` check.

  Detects calls to standard library iterator algorithms that could be replaced
  with a ranges version instead.

- New :doc:`modernize-use-std-format
  <clang-tidy/checks/modernize/use-std-format>` check.

  Converts calls to ``absl::StrFormat``, or other functions via
  configuration options, to C++20's ``std::format``, or another function
  via a configuration option, modifying the format string appropriately and
  removing now-unnecessary calls to ``std::string::c_str()`` and
  ``std::string::data()``.

- New :doc:`readability-enum-initial-value
  <clang-tidy/checks/readability/enum-initial-value>` check.

  Enforces consistent style for enumerators' initialization, covering three
  styles: none, first only, or all initialized explicitly.

- New :doc:`readability-math-missing-parentheses
  <clang-tidy/checks/readability/math-missing-parentheses>` check.

  Check for missing parentheses in mathematical expressions that involve
  operators of different priorities.

- New :doc:`readability-use-std-min-max
  <clang-tidy/checks/readability/use-std-min-max>` check.

  Replaces certain conditional statements with equivalent calls to
  ``std::min`` or ``std::max``.


New check aliases
^^^^^^^^^^^^^^^^^

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`readability-redundant-smartptr-get
  <clang-tidy/checks/readability/redundant-smartptr-get>` check to
  remove `->`, when reduntant `get()` is removed.

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
