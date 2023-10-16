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

- The extract variable tweak gained support for extracting lambda expressions to a variable.

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

- Improved :program:`clang-tidy-diff.py` script. It now returns exit code `1`
  if any :program:`clang-tidy` subprocess exits with a non-zero code or if
  exporting fixes fails.

New checks
^^^^^^^^^^

- New :doc:`bugprone-compare-pointer-to-member-virtual-function
  <clang-tidy/checks/bugprone/compare-pointer-to-member-virtual-function>` check.

  Detects equality comparison between pointer to member virtual function and
  anything other than null-pointer-constant.

- New :doc:`bugprone-inc-dec-in-conditions
  <clang-tidy/checks/bugprone/inc-dec-in-conditions>` check.

  Detects when a variable is both incremented/decremented and referenced inside
  a complex condition and suggests moving them outside to avoid ambiguity in
  the variable's value.

- New :doc:`bugprone-incorrect-enable-if
  <clang-tidy/checks/bugprone/incorrect-enable-if>` check.

  Detects incorrect usages of ``std::enable_if`` that don't name the nested
  ``type`` type.

- New :doc:`bugprone-multi-level-implicit-pointer-conversion
  <clang-tidy/checks/bugprone/multi-level-implicit-pointer-conversion>` check.

  Detects implicit conversions between pointers of different levels of
  indirection.

- New :doc:`bugprone-optional-value-conversion
  <clang-tidy/checks/bugprone/optional-value-conversion>` check.

  Detects potentially unintentional and redundant conversions where a value is
  extracted from an optional-like type and then used to create a new instance
  of the same optional-like type.

- New :doc:`cppcoreguidelines-no-suspend-with-lock
  <clang-tidy/checks/cppcoreguidelines/no-suspend-with-lock>` check.

  Flags coroutines that suspend while a lock guard is in scope at the
  suspension point.

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

- New alias :doc:`cppcoreguidelines-macro-to-enum
  <clang-tidy/checks/cppcoreguidelines/macro-to-enum>` to :doc:`modernize-macro-to-enum
  <clang-tidy/checks/modernize/macro-to-enum>` was added.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-lambda-function-name
  <clang-tidy/checks/bugprone/lambda-function-name>` check by adding option
  `IgnoreMacros` to ignore warnings in macros.

- Improved :doc:`bugprone-non-zero-enum-to-bool-conversion
  <clang-tidy/checks/bugprone/non-zero-enum-to-bool-conversion>` check by
  eliminating false positives resulting from direct usage of bitwise operators.

- Improved :doc:`bugprone-reserved-identifier
  <clang-tidy/checks/bugprone/reserved-identifier>` check, so that it does not
  warn on macros starting with underscore and lowercase letter.

- Improved :doc:`bugprone-undefined-memory-manipulation
  <clang-tidy/checks/bugprone/undefined-memory-manipulation>` check to support
  fixed-size arrays of non-trivial types.

- Improved :doc:`cppcoreguidelines-avoid-non-const-global-variables
  <clang-tidy/checks/cppcoreguidelines/avoid-non-const-global-variables>` check
  to ignore ``static`` variables declared within the scope of
  ``class``/``struct``.

- Improved :doc:`cppcoreguidelines-avoid-reference-coroutine-parameters
  <clang-tidy/checks/cppcoreguidelines/avoid-reference-coroutine-parameters>`
  check to ignore false positives related to matching parameters of non
  coroutine functions and increase issue detection for cases involving type
  aliases with references.

- Improved :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines/prefer-member-initializer>` check to
  ignore delegate constructors.

- Improved :doc:`cppcoreguidelines-pro-bounds-array-to-pointer-decay
  <clang-tidy/checks/cppcoreguidelines/pro-bounds-array-to-pointer-decay>` check
  to ignore predefined expression (e.g., ``__func__``, ...).

- Improved :doc:`cppcoreguidelines-pro-bounds-constant-array-index
  <clang-tidy/checks/cppcoreguidelines/pro-bounds-constant-array-index>` check
  to perform checks on derived classes of  ``std::array``.

- Improved :doc:`cppcoreguidelines-pro-type-member-init
  <clang-tidy/checks/cppcoreguidelines/pro-type-member-init>` check to ignore
  dependent delegate constructors.

- Improved :doc:`cppcoreguidelines-pro-type-vararg
  <clang-tidy/checks/cppcoreguidelines/pro-type-vararg>` check to ignore
  false-positives in unevaluated context (e.g., ``decltype``, ``sizeof``, ...).

- Improved :doc:`llvm-namespace-comment
  <clang-tidy/checks/llvm/namespace-comment>` check to provide fixes for
  ``inline`` namespaces in the same format as :program:`clang-format`.

- Improved :doc:`llvmlibc-callee-namespace
  <clang-tidy/checks/llvmlibc/callee-namespace>` to support
  customizable namespace. This matches the change made to implementation in
  namespace.

- Improved :doc:`llvmlibc-implementation-in-namespace
  <clang-tidy/checks/llvmlibc/implementation-in-namespace>` to support
  customizable namespace. This further allows for testing the libc when the
  system-libc is also LLVM's libc.

- Improved :doc:`misc-const-correctness
  <clang-tidy/checks/misc/const-correctness>` check to avoid false positive when
  using pointer to member function.

- Improved :doc:`misc-include-cleaner
  <clang-tidy/checks/misc/include-cleaner>` check by adding option
  `DeduplicateFindings` to output one finding per symbol occurrence, avoid
  inserting the same header multiple times, fix a bug where `IgnoreHeaders`
  option won't work with verbatim/std headers.

- Improved :doc:`misc-redundant-expression
  <clang-tidy/checks/misc/redundant-expression>` check to ignore
  false-positives in unevaluated context (e.g., ``decltype``).

- Improved :doc:`modernize-loop-convert
  <clang-tidy/checks/modernize/loop-convert>` to support for-loops with
  iterators initialized by free functions like ``begin``, ``end``, or ``size``.

- Improved :doc:`modernize-use-equals-delete
  <clang-tidy/checks/modernize/use-equals-delete>` check to ignore
  false-positives when special member function is actually used or implicit.

- Improved :doc:`modernize-use-nullptr
  <clang-tidy/checks/modernize/use-nullptr>` check by adding option
  `IgnoredTypes` that can be used to exclude some pointer types.

- Improved :doc:`modernize-use-std-print
  <clang-tidy/checks/modernize/use-std-print>` check to accurately generate
  fixes for reordering arguments.

- Improved :doc:`modernize-use-using
  <clang-tidy/checks/modernize/use-using>` check to fix function pointer and
  forward declared ``typedef`` correctly.

- Improved :doc:`performance-faster-string-find
  <clang-tidy/checks/performance/faster-string-find>` check to properly escape
  single quotes.

- Improved :doc:`performance-noexcept-swap
  <clang-tidy/checks/performance/noexcept-swap>` check to enforce a stricter
  match with the swap function signature, eliminating false-positives.

- Improved :doc:`readability-braces-around-statements
  <clang-tidy/checks/readability/braces-around-statements>` check to
  ignore false-positive for ``if constexpr`` in lambda expression.

- Improved :doc:`readability-const-params-in-decls
  <clang-tidy/checks/readability/const-params-in-decls>` place the hint check under
  the const token instead of the beginning of parameter declaration.

- Improved :doc:`readability-container-size-empty
  <clang-tidy/checks/readability/container-size-empty>` check to
  detect comparison between string and empty string literals and support
  ``length()`` method as an alternative to ``size()``.

- Improved :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check to issue accurate
  warnings when a type's forward declaration precedes its definition.
  Additionally, it now provides appropriate warnings for ``struct`` and
  ``union`` in C, while also incorporating support for the
  ``Leading_upper_snake_case`` naming convention. The handling of ``typedef``
  has been enhanced, particularly within complex types like function pointers
  and cases where style checks were omitted when functions started with macros.

- Improved :doc:`readability-implicit-bool-conversion
  <clang-tidy/checks/readability/implicit-bool-conversion>` check to take
  do-while loops into account for the `AllowIntegerConditions` and
  `AllowPointerConditions` options.

- Improved :doc:`readability-non-const-parameter
  <clang-tidy/checks/readability/non-const-parameter>` check to ignore
  false-positives in initializer list of record.

- Improved :doc:`readability-static-accessed-through-instance
  <clang-tidy/checks/readability/static-accessed-through-instance>` check to
  identify calls to static member functions with out-of-class inline definitions.


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
