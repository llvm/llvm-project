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
- A new tweak was added for turning unscoped into scoped enums.

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

- Improved :program:`clang-tidy-diff.py` script.
    * Return exit code `1` if any :program:`clang-tidy` subprocess exits with
      a non-zero code or if exporting fixes fails.

    * Accept a directory as a value for `-export-fixes` to export individual
      yaml files for each compilation unit.

    * Introduce a `-config-file` option that forwards a configuration file to
      :program:`clang-tidy`. Corresponds to the `--config-file` option in
      :program:`clang-tidy`.

- Improved :program:`run-clang-tidy.py` script. It now accepts a directory
  as a value for `-export-fixes` to export individual yaml files for each
  compilation unit.


New checks
^^^^^^^^^^

- New :doc:`bugprone-casting-through-void
  <clang-tidy/checks/bugprone/casting-through-void>` check.

  Detects unsafe or redundant two-step casting operations involving ``void*``.

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

- New :doc:`bugprone-unused-local-non-trivial-variable
  <clang-tidy/checks/bugprone/unused-local-non-trivial-variable>` check.

  Warns when a local non trivial variable is unused within a function.

- New :doc:`cppcoreguidelines-no-suspend-with-lock
  <clang-tidy/checks/cppcoreguidelines/no-suspend-with-lock>` check.

  Flags coroutines that suspend while a lock guard is in scope at the
  suspension point.

- New :doc:`hicpp-ignored-remove-result
  <clang-tidy/checks/hicpp/ignored-remove-result>` check.

  Ensure that the result of ``std::remove``, ``std::remove_if`` and
  ``std::unique`` are not ignored according to rule 17.5.1.

- New :doc:`misc-coroutine-hostile-raii
  <clang-tidy/checks/misc/coroutine-hostile-raii>` check.

  Detects when objects of certain hostile RAII types persists across suspension
  points in a coroutine. Such hostile types include scoped-lockable types and
  types belonging to a configurable denylist.

- New :doc:`modernize-use-constraints
  <clang-tidy/checks/modernize/use-constraints>` check.

  Replace ``enable_if`` with C++20 requires clauses.

- New :doc:`modernize-use-starts-ends-with
  <clang-tidy/checks/modernize/use-starts-ends-with>` check.

  Checks whether a ``find`` or ``rfind`` result is compared with 0 and suggests
  replacing with ``starts_with`` when the method exists in the class. Notably,
  this will work with ``std::string`` and ``std::string_view``.

- New :doc:`modernize-use-std-numbers
  <clang-tidy/checks/modernize/use-std-numbers>` check.

  Finds constants and function calls to math functions that can be replaced
  with C++20's mathematical constants from the ``numbers`` header and
  offers fix-it hints.

- New :doc:`performance-enum-size
  <clang-tidy/checks/performance/enum-size>` check.

  Recommends the smallest possible underlying type for an ``enum`` or ``enum``
  class based on the range of its enumerators.

- New :doc:`readability-avoid-nested-conditional-operator
  <clang-tidy/checks/readability/avoid-nested-conditional-operator>` check.

  Identifies instances of nested conditional operators in the code.

- New :doc:`readability-avoid-return-with-void-value
  <clang-tidy/checks/readability/avoid-return-with-void-value>` check.

  Finds return statements with ``void`` values used within functions with
  ``void`` result types.

- New :doc:`readability-redundant-casting
  <clang-tidy/checks/readability/redundant-casting>` check.

  Detects explicit type casting operations that involve the same source and
  destination types, and subsequently recommend their removal.
  
- New :doc:`readability-redundant-inline-specifier
  <clang-tidy/checks/readability/redundant-inline-specifier>` check.

  Detects redundant ``inline`` specifiers on function and variable declarations.

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

- Improved :doc:`abseil-string-find-startswith
  <clang-tidy/checks/abseil/string-find-startswith>` check to also consider
  ``std::basic_string_view`` in addition to ``std::basic_string`` by default.

- Improved :doc:`bugprone-assert-side-effect
  <clang-tidy/checks/bugprone/assert-side-effect>` check to report usage of
  non-const ``<<`` and ``>>`` operators in assertions and fixed some false-positives
  with const operators.

- Improved :doc:`bugprone-dangling-handle
  <clang-tidy/checks/bugprone/dangling-handle>` check to support functional
  casting during type conversions at variable initialization, now with improved
  compatibility for C++17 and later versions.

- Improved :doc:`bugprone-exception-escape
  <clang-tidy/checks/bugprone/exception-escape>` check by extending the default
  check function names to include ``iter_swap`` and ``iter_move``.

- Improved :doc:`bugprone-implicit-widening-of-multiplication-result
  <clang-tidy/checks/bugprone/implicit-widening-of-multiplication-result>` check
  to correctly emit fixes.

- Improved :doc:`bugprone-lambda-function-name
  <clang-tidy/checks/bugprone/lambda-function-name>` check by adding option
  `IgnoreMacros` to ignore warnings in macros.

- Improved :doc:`bugprone-non-zero-enum-to-bool-conversion
  <clang-tidy/checks/bugprone/non-zero-enum-to-bool-conversion>` check by
  eliminating false positives resulting from direct usage of bitwise operators.

- Improved :doc:`bugprone-reserved-identifier
  <clang-tidy/checks/bugprone/reserved-identifier>` check, so that it does not
  warn on macros starting with underscore and lowercase letter.

- Improved :doc:`bugprone-sizeof-expression
  <clang-tidy/checks/bugprone/sizeof-expression>` check diagnostics to precisely
  highlight specific locations, providing more accurate guidance.

- Improved :doc:`bugprone-unchecked-optional-access
  <clang-tidy/checks/bugprone/unchecked-optional-access>` check, so that it does
  not crash during handling of optional values.

- Improved :doc:`bugprone-undefined-memory-manipulation
  <clang-tidy/checks/bugprone/undefined-memory-manipulation>` check to support
  fixed-size arrays of non-trivial types.

- Improved :doc:`bugprone-unused-return-value
  <clang-tidy/checks/bugprone/unused-return-value>` check diagnostic message,
  added support for detection of unused results when cast to non-``void`` type.
  Casting to ``void`` no longer suppresses issues by default, control this
  behavior with the new `AllowCastToVoid` option.

- Improved :doc:`cppcoreguidelines-avoid-non-const-global-variables
  <clang-tidy/checks/cppcoreguidelines/avoid-non-const-global-variables>` check
  to ignore ``static`` variables declared within the scope of
  ``class``/``struct``.

- Improved :doc:`cppcoreguidelines-avoid-reference-coroutine-parameters
  <clang-tidy/checks/cppcoreguidelines/avoid-reference-coroutine-parameters>`
  check to ignore false positives related to matching parameters of non
  coroutine functions and increase issue detection for cases involving type
  aliases with references.

- Improved :doc:`cppcoreguidelines-missing-std-forward
  <clang-tidy/checks/cppcoreguidelines/missing-std-forward>` check to
  address false positives in the capture list and body of lambdas.

- Improved :doc:`cppcoreguidelines-narrowing-conversions
  <clang-tidy/checks/cppcoreguidelines/narrowing-conversions>` check by
  extending the `IgnoreConversionFromTypes` option to include types without a
  declaration, such as built-in types.

- Improved :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines/prefer-member-initializer>` check to
  ignore delegate constructors and ignore re-assignment for reference or when
  initialization depend on field that is initialized before.

- Improved :doc:`cppcoreguidelines-pro-bounds-array-to-pointer-decay
  <clang-tidy/checks/cppcoreguidelines/pro-bounds-array-to-pointer-decay>` check
  to ignore predefined expression (e.g., ``__func__``, ...).

- Improved :doc:`cppcoreguidelines-pro-bounds-constant-array-index
  <clang-tidy/checks/cppcoreguidelines/pro-bounds-constant-array-index>` check
  to perform checks on derived classes of  ``std::array``.

- Improved :doc:`cppcoreguidelines-pro-type-const-cast
  <clang-tidy/checks/cppcoreguidelines/pro-type-const-cast>` check to ignore
  casts to ``const`` or ``volatile`` type (controlled by `StrictMode` option)
  and casts in implicitly invoked code.

- Improved :doc:`cppcoreguidelines-pro-type-member-init
  <clang-tidy/checks/cppcoreguidelines/pro-type-member-init>` check to ignore
  dependent delegate constructors.

- Improved :doc:`cppcoreguidelines-pro-type-static-cast-downcast
  <clang-tidy/checks/cppcoreguidelines/pro-type-static-cast-downcast>` check to
  disregard casts on non-polymorphic types when the `StrictMode` option is set
  to `false`.

- Improved :doc:`cppcoreguidelines-pro-type-vararg
  <clang-tidy/checks/cppcoreguidelines/pro-type-vararg>` check to ignore
  false-positives in unevaluated context (e.g., ``decltype``, ``sizeof``, ...).

- Improved :doc:`cppcoreguidelines-rvalue-reference-param-not-moved
  <clang-tidy/checks/cppcoreguidelines/rvalue-reference-param-not-moved>` check
  to ignore unused parameters when they are marked as unused and parameters of
  deleted functions and constructors.

- Improved :doc:`google-readability-casting
  <clang-tidy/checks/google/readability-casting>` check to ignore constructor
  calls disguised as functional casts.

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

- Improved :doc:`llvmlibc-inline-function-decl
  <clang-tidy/checks/llvmlibc/inline-function-decl>` to properly ignore implicit
  functions, such as struct constructors, and explicitly deleted functions.

- Improved :doc:`misc-const-correctness
  <clang-tidy/checks/misc/const-correctness>` check to avoid false positive when
  using pointer to member function. Additionally, the check no longer emits
  a diagnostic when a variable that is not type-dependent is an operand of a
  type-dependent binary operator. Improved performance of the check through
  optimizations.

- Improved :doc:`misc-include-cleaner
  <clang-tidy/checks/misc/include-cleaner>` check by adding option
  `DeduplicateFindings` to output one finding per symbol occurrence, avoid
  inserting the same header multiple times, fix a bug where `IgnoreHeaders`
  option won't work with verbatim/std headers.

- Improved :doc:`misc-redundant-expression
  <clang-tidy/checks/misc/redundant-expression>` check to ignore
  false-positives in unevaluated context (e.g., ``decltype``).

- Improved :doc:`misc-static-assert
  <clang-tidy/checks/misc/static-assert>` check to ignore false-positives when
  referring to non-``constexpr`` variables in non-unevaluated context.

- Improved :doc:`misc-unused-using-decls
  <clang-tidy/checks/misc/unused-using-decls>` check to avoid false positive when
  using in elaborated type and only check C++ files.

- Improved :doc:`modernize-avoid-bind
  <clang-tidy/checks/modernize/avoid-bind>` check to
  not emit a ``return`` for fixes when the function returns ``void`` and to
  provide valid fixes for cases involving bound C++ operators.

- Improved :doc:`modernize-loop-convert
  <clang-tidy/checks/modernize/loop-convert>` to support for-loops with
  iterators initialized by free functions like ``begin``, ``end``, or ``size``
  and avoid crash for array of dependent array and non-dereferenceable builtin
  types used as iterators.

- Improved :doc:`modernize-make-shared
  <clang-tidy/checks/modernize/make-shared>` check to support
  ``std::shared_ptr`` implementations that inherit the ``reset`` method from a
  base class.

- Improved :doc:`modernize-return-braced-init-list
  <clang-tidy/checks/modernize/return-braced-init-list>` check to ignore
  false-positives when constructing the container with ``count`` copies of
  elements with value ``value``.

- Improved :doc:`modernize-use-auto
  <clang-tidy/checks/modernize/use-auto>` to avoid create incorrect fix hints
  for pointer to array type and pointer to function type.

- Improved :doc:`modernize-use-emplace
  <clang-tidy/checks/modernize/use-emplace>` to not replace aggregates that
  ``emplace`` cannot construct with aggregate initialization.

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
  forward declared ``typedef`` correctly. Added option `IgnoreExternC` to ignore
  ``typedef`` declaration in ``extern "C"`` scope.

- Improved :doc:`performance-faster-string-find
  <clang-tidy/checks/performance/faster-string-find>` check to properly escape
  single quotes.

- Improved :doc:`performance-for-range-copy
  <clang-tidy/checks/performance/for-range-copy>` check to handle cases where
  the loop variable is a structured binding.

- Improved :doc:`performance-noexcept-move-constructor
  <clang-tidy/checks/performance/noexcept-move-constructor>` to better handle
  conditional ``noexcept`` expressions, eliminating false-positives.

- Improved :doc:`performance-noexcept-swap
  <clang-tidy/checks/performance/noexcept-swap>` check to enforce a stricter
  match with the swap function signature and better handling of condition
  ``noexcept`` expressions, eliminating false-positives. ``iter_swap`` function
  name is checked by default.

- Improved :doc:`readability-braces-around-statements
  <clang-tidy/checks/readability/braces-around-statements>` check to
  ignore false-positive for ``if constexpr`` in lambda expression.

- Improved :doc:`readability-avoid-const-params-in-decls
  <clang-tidy/checks/readability/avoid-const-params-in-decls>` diagnostics to
  highlight the ``const`` location

- Improved :doc:`readability-container-contains
  <clang-tidy/checks/readability/container-contains>` to correctly handle
  integer literals with suffixes in fix-its.

- Improved :doc:`readability-container-size-empty
  <clang-tidy/checks/readability/container-size-empty>` check to
  detect comparison between string and empty string literals and support
  ``length()`` method as an alternative to ``size()``. Resolved false positives
  tied to negative values from size-like methods, and one triggered by size
  checks below zero.

- Improved :doc:`readability-function-size
  <clang-tidy/checks/readability/function-size>` check configuration to use
  `none` rather than `-1` to disable some parameters.

- Improved :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check to issue accurate
  warnings when a type's forward declaration precedes its definition.
  Additionally, it now provides appropriate warnings for ``struct`` and
  ``union`` in C, while also incorporating support for the
  ``Leading_upper_snake_case`` naming convention. The handling of ``typedef``
  has been enhanced, particularly within complex types like function pointers
  and cases where style checks were omitted when functions started with macros.
  Added support for C++20 ``concept`` declarations. ``Camel_Snake_Case`` and
  ``camel_Snake_Case`` now detect more invalid identifier names. Fields in
  anonymous records (i.e. anonymous structs and unions) now can be checked with
  the naming rules associated with their enclosing scopes rather than the naming
  rules of public ``struct``/``union`` members.

- Improved :doc:`readability-implicit-bool-conversion
  <clang-tidy/checks/readability/implicit-bool-conversion>` check to take
  do-while loops into account for the `AllowIntegerConditions` and
  `AllowPointerConditions` options. It also now provides more consistent
  suggestions when parentheses are added to the return value or expressions.
  It also ignores false-positives for comparison containing bool bitfield.

- Improved :doc:`readability-misleading-indentation
  <clang-tidy/checks/readability/misleading-indentation>` check to ignore
  false-positives for line started with empty macro.

- Improved :doc:`readability-non-const-parameter
  <clang-tidy/checks/readability/non-const-parameter>` check to ignore
  false-positives in initializer list of record.

- Improved :doc:`readability-redundant-member-init
  <clang-tidy/checks/readability/redundant-member-init>` check to now also
  detect redundant in-class initializers.

- Improved :doc:`readability-simplify-boolean-expr
  <clang-tidy/checks/readability/simplify-boolean-expr>` check by adding the
  new option `IgnoreMacros` that allows to ignore boolean expressions originating
  from expanded macros.

- Improved :doc:`readability-simplify-subscript-expr
  <clang-tidy/checks/readability/simplify-subscript-expr>` check by extending
  the default value of the `Types` option to include ``std::span``.

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
