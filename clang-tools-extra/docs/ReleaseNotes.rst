.. If you want to modify sections/contents permanently, you should modify both
   ReleaseNotes.rst and ReleaseNotesTemplate.txt.

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

Potentially Breaking Changes
----------------------------

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

Improvements to clang-tidy
--------------------------

- Improved :program:`check_clang_tidy.py` script by adding the `-check-header`
  argument to simplify testing of header files. This argument automatically
  manages the creation of temporary header files and ensures that diagnostics
  and fixes are verified for the specified headers.

New checks
^^^^^^^^^^

- New :doc:`bugprone-unsafe-to-allow-exceptions
  <clang-tidy/checks/bugprone/unsafe-to-allow-exceptions>` check.

  Finds functions where throwing exceptions is unsafe but the function is still
  marked as potentially throwing.

- New :doc:`llvm-type-switch-case-types
  <clang-tidy/checks/llvm/type-switch-case-types>` check.

  Finds ``llvm::TypeSwitch::Case`` calls with redundant explicit template
  arguments that can be inferred from the lambda parameter type.

- New :doc:`llvm-use-vector-utils
  <clang-tidy/checks/llvm/use-vector-utils>` check.

  Finds calls to ``llvm::to_vector(llvm::map_range(...))`` and
  ``llvm::to_vector(llvm::make_filter_range(...))`` that can be replaced with
  ``llvm::map_to_vector`` and ``llvm::filter_to_vector``.

- New :doc:`modernize-use-string-view
  <clang-tidy/checks/modernize/use-string-view>` check.

  Looks for functions returning ``std::[w|u8|u16|u32]string`` and suggests to
  change it to ``std::[...]string_view`` for performance reasons if possible.

- New :doc:`modernize-use-structured-binding
  <clang-tidy/checks/modernize/use-structured-binding>` check.

  Finds places where structured bindings could be used to decompose pairs and
  suggests replacing them.

- New :doc:`performance-string-view-conversions
  <clang-tidy/checks/performance/string-view-conversions>` check.

  Finds and removes redundant conversions from ``std::[w|u8|u16|u32]string_view`` to
  ``std::[...]string`` in call expressions expecting ``std::[...]string_view``.

- New :doc:`readability-trailing-comma
  <clang-tidy/checks/readability/trailing-comma>` check.

  Checks for presence or absence of trailing commas in enum definitions and
  initializer lists.

- New :doc:`modernize-replace-with-std-copy
  <clang-tidy/checks/modernize-replace-with-std-copy>` check.

  Tries to replace calls to ``memmove`` and ``memcpy`` with an equivalent call to ``std::copy_n``.

New check aliases
^^^^^^^^^^^^^^^^^

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-argument-comment
  <clang-tidy/checks/bugprone/argument-comment>` to also check for C++11
  inherited constructors.

- Improved :doc:`bugprone-bad-signal-to-kill-thread
  <clang-tidy/checks/bugprone/bad-signal-to-kill-thread>` check by fixing false
  negatives when the ``SIGTERM`` macro is obtained from a precompiled header.

- Improved :doc:`bugprone-exception-escape
  <clang-tidy/checks/bugprone/exception-escape>` check by adding
  `TreatFunctionsWithoutSpecificationAsThrowing` option to support reporting
  for unannotated functions, enabling reporting when no explicit ``throw``
  is seen and allowing separate tuning for known and unknown implementations.

- Improved :doc:`bugprone-macro-parentheses
  <clang-tidy/checks/bugprone/macro-parentheses>` check by printing the macro
  definition in the warning message if the macro is defined on command line.

- Improved :doc:`bugprone-string-constructor
  <clang-tidy/checks/bugprone/string-constructor>` check to detect suspicious
  string constructor calls when the string class constructor has a default
  allocator argument.

- Improved :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` check by adding the function
  ``std::get_temporary_buffer`` to the default list of unsafe functions. (This
  function is unsafe, useless, deprecated in C++17 and removed in C++20).

- Improved :doc:`bugprone-use-after-move
  <clang-tidy/checks/bugprone/use-after-move>` check by including the name of
  the invalidating function in the warning message when a custom invalidation
  function is used (via the `InvalidationFunctions` option).

- Improved :doc:`cppcoreguidelines-init-variables
  <clang-tidy/checks/cppcoreguidelines/init-variables>` check by ensuring that
  member pointers are correctly flagged as uninitialized.

- Improved :doc:`cppcoreguidelines-pro-type-vararg
  <clang-tidy/checks/cppcoreguidelines/pro-type-vararg>` check by no longer
  warning on builtins with custom type checking (e.g., type-generic builtins
  like ``__builtin_clzg``) that use variadic declarations as an implementation
  detail.

- Improved :doc:`llvm-use-ranges
  <clang-tidy/checks/llvm/use-ranges>` check by adding support for the following
  algorithms: ``std::accumulate``, ``std::replace_copy``, and
  ``std::replace_copy_if``.

- Improved :doc:`misc-const-correctness
  <clang-tidy/checks/misc/const-correctness>` check:

  - Added support for analyzing function parameters with the `AnalyzeParameters`
    option.

- Improved :doc:`modernize-pass-by-value
  <clang-tidy/checks/modernize/pass-by-value>` check by adding `IgnoreMacros`
  option to suppress warnings in macros.

- Improved :doc:`modernize-redundant-void-arg
  <clang-tidy/checks/modernize/redundant-void-arg>` check to work in C23.

- Improved :doc:`modernize-use-std-format
  <clang-tidy/checks/modernize/use-std-format>` check by fixing a crash
  when an argument is part of a macro expansion.

- Improved :doc:`modernize-use-using
  <clang-tidy/checks/modernize/use-using>` check by avoiding the generation
  of invalid code for function types with redundant parentheses.

- Improved :doc:`performance-enum-size
  <clang-tidy/checks/performance/enum-size>` check:

  - Exclude ``enum`` in ``extern "C"`` blocks.

  - Improved the ignore list to correctly handle ``typedef`` and  ``enum``.

- Improved :doc:`performance-inefficient-vector-operation
  <clang-tidy/checks/performance/inefficient-vector-operation>` check by
  correctly handling vector-like classes when ``push_back``/``emplace_back`` are
  inherited.

- Improved :doc:`performance-move-const-arg
  <clang-tidy/checks/performance/move-const-arg>` check by avoiding false
  positives on trivially copyable types with a non-public copy constructor.

- Improved :doc:`readability-container-size-empty
  <clang-tidy/checks/readability/container-size-empty>` check by fixing a crash
  when a member expression has a non-identifier name.

- Improved :doc:`readability-enum-initial-value
  <clang-tidy/checks/readability/enum-initial-value>` check: the warning message
  now uses separate note diagnostics for each uninitialized enumerator, making
  it easier to see which specific enumerators need explicit initialization.

- Improved :doc:`readability-non-const-parameter
  <clang-tidy/checks/readability/non-const-parameter>` check by avoiding false
  positives on parameters used in dependent expressions (e.g. inside generic
  lambdas).

- Improved :doc:`readability-simplify-boolean-expr
  <clang-tidy/checks/readability/simplify-boolean-expr>` check to provide valid
  fix suggestions for C23 and later by not using ``static_cast``.

- Improved :doc:`readability-suspicious-call-argument
  <clang-tidy/checks/readability/suspicious-call-argument>` check by avoiding a
  crash from invalid ``Abbreviations`` option.

Removed checks
^^^^^^^^^^^^^^

Miscellaneous
^^^^^^^^^^^^^

Improvements to include-fixer
-----------------------------

Improvements to clang-include-fixer
-----------------------------------

Improvements to modularize
--------------------------

Improvements to pp-trace
------------------------

Clang-tidy Visual Studio plugin
-------------------------------
