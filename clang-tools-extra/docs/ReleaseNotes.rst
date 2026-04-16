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

- Deprecated the :program:`clang-tidy` check :doc:`performance-faster-string-find
  <clang-tidy/checks/performance/faster-string-find>`. It has been renamed to
  :doc:`performance-prefer-single-char-overloads
  <clang-tidy/checks/performance/prefer-single-char-overloads>`.
  The original check will be removed in the 25th release.

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

- Now also provides include files without extension, if they are in a directory
  only called ``include``.

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

- Improved :program:`clang-tidy` ``-store-check-profile`` by generating valid
  JSON when the source file path contains characters that require JSON escaping.

New checks
^^^^^^^^^^

- New :doc:`bugprone-assignment-in-selection-statement
  <clang-tidy/checks/bugprone/assignment-in-selection-statement>` check.

  Finds assignments within selection statements.

- New :doc:`bugprone-unsafe-to-allow-exceptions
  <clang-tidy/checks/bugprone/unsafe-to-allow-exceptions>` check.

  Finds functions where throwing exceptions is unsafe but the function is still
  marked as potentially throwing.

- New :doc:`llvm-redundant-casting
  <clang-tidy/checks/llvm/redundant-casting>` check.

  Points out uses of ``cast<>``, ``dyn_cast<>`` and their ``or_null`` variants
  that are unnecessary because the argument already is of the target type, or a
  derived type thereof.

- New :doc:`llvm-type-switch-case-types
  <clang-tidy/checks/llvm/type-switch-case-types>` check.

  Finds ``llvm::TypeSwitch::Case`` calls with redundant explicit template
  arguments that can be inferred from the lambda parameter type.

- New :doc:`llvm-use-vector-utils
  <clang-tidy/checks/llvm/use-vector-utils>` check.

  Finds calls to ``llvm::to_vector(llvm::map_range(...))`` and
  ``llvm::to_vector(llvm::make_filter_range(...))`` that can be replaced with
  ``llvm::map_to_vector`` and ``llvm::filter_to_vector``.

- New :doc:`modernize-use-std-bit
  <clang-tidy/checks/modernize/use-std-bit>` check.

  Finds common idioms which can be replaced by standard functions from the
  ``<bit>`` C++20 header.

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

- New :doc:`performance-use-std-move
  <clang-tidy/checks/performance/use-std-move>` check.

  Suggests insertion of ``std::move(...)`` to turn copy assignment operator
  calls into move assignment ones, when deemed valid and profitable.

- New :doc:`readability-redundant-qualified-alias
  <clang-tidy/checks/readability/redundant-qualified-alias>` check.

  Finds redundant identity type aliases that re-expose a qualified name and can
  be replaced with a ``using`` declaration.

- New :doc:`readability-trailing-comma
  <clang-tidy/checks/readability/trailing-comma>` check.

  Checks for presence or absence of trailing commas in enum definitions and
  initializer lists.

New check aliases
^^^^^^^^^^^^^^^^^

- New alias :doc:`cert-exp45-c <clang-tidy/checks/cert/exp45-c>`
  to :doc:`bugprone-assignment-in-selection-statement
  <clang-tidy/checks/bugprone/assignment-in-selection-statement>`.

- Renamed :doc:`hicpp-exception-baseclass
  <clang-tidy/checks/hicpp/exception-baseclass>`
  to :doc:`bugprone-std-exception-baseclass
  <clang-tidy/checks/bugprone/std-exception-baseclass>`.
  The `hicpp-exception-baseclass` name is kept as an alias.

- Renamed :doc:`hicpp-ignored-remove-result
  <clang-tidy/checks/hicpp/ignored-remove-result>`
  to :doc:`bugprone-unused-return-value
  <clang-tidy/checks/bugprone/unused-return-value>`.
  The `hicpp-ignored-remove-result` name is kept as an alias.

- Renamed :doc:`hicpp-no-assembler <clang-tidy/checks/hicpp/no-assembler>`
  to :doc:`portability-no-assembler
  <clang-tidy/checks/portability/no-assembler>`. The `hicpp-no-assembler`
  name is kept as an alias.

- Renamed :doc:`performance-faster-string-find
  <clang-tidy/checks/performance/faster-string-find>` to
  :doc:`performance-prefer-single-char-overloads
  <clang-tidy/checks/performance/prefer-single-char-overloads>`.
  The `performance-faster-string-find` name is kept as an alias.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-argument-comment
  <clang-tidy/checks/bugprone/argument-comment>` to also check for C++11
  inherited constructors.

- Improved :doc:`bugprone-bad-signal-to-kill-thread
  <clang-tidy/checks/bugprone/bad-signal-to-kill-thread>` check by fixing false
  negatives when the ``SIGTERM`` macro is obtained from a precompiled header.

- Improved :doc:`bugprone-casting-through-void
  <clang-tidy/checks/bugprone/casting-through-void>` check by running only on
  C++ files because suggested ``reinterpret_cast`` is not available in pure C.

- Improved :doc:`bugprone-derived-method-shadowing-base-method
  <clang-tidy/checks/bugprone/derived-method-shadowing-base-method>` check by
  correctly ignoring function templates.

- Improved :doc:`bugprone-exception-escape
  <clang-tidy/checks/bugprone/exception-escape>` check by adding
  `TreatFunctionsWithoutSpecificationAsThrowing` option to support reporting
  for unannotated functions, enabling reporting when no explicit ``throw``
  is seen and allowing separate tuning for known and unknown implementations.

- Improved :doc:`bugprone-fold-init-type
  <clang-tidy/checks/bugprone/fold-init-type>` check by detecting precision
  loss in overloads with transparent standard functors (e.g. ``std::plus<>``)
  for ``std::accumulate``, ``std::reduce``, and ``std::inner_product``.

- Improved :doc:`bugprone-inc-dec-in-conditions
  <clang-tidy/checks/bugprone/inc-dec-in-conditions>` check by fixing a false
  positive when increment/decrement operators appear inside lambda bodies that
  are part of a condition expression.

- Improved :doc:`bugprone-incorrect-enable-if
  <clang-tidy/checks/bugprone/incorrect-enable-if>` check to not
  insert an extraneous ``typename`` on code like
  ``typename std::enable_if<...>``, where there's already a ``typename`` and
  only the ``::type`` at the end is missing.

- Improved :doc:`bugprone-macro-parentheses
  <clang-tidy/checks/bugprone/macro-parentheses>` check by printing the macro
  definition in the warning message if the macro is defined on command line.

- Improved :doc:`bugprone-pointer-arithmetic-on-polymorphic-object
  <clang-tidy/checks/bugprone/pointer-arithmetic-on-polymorphic-object>` check
  by fixing a false positive when ``operator[]`` is used in a dependent context.

- Improved :doc:`bugprone-std-namespace-modification
  <clang-tidy/checks/bugprone/std-namespace-modification>` check by fixing
  false positives when extending the standard library with a specialization of
  user-defined type and by removing detection of the compiler generated ``std``
  namespace extensions.

- Improved :doc:`bugprone-string-constructor
  <clang-tidy/checks/bugprone/string-constructor>` check to detect suspicious
  string constructor calls when the string class constructor has a default
  allocator argument.

- Improved :doc:`bugprone-unchecked-optional-access
  <clang-tidy/checks/bugprone/unchecked-optional-access>` to recognize common
  GoogleTest macros such as ``ASSERT_TRUE`` and ``ASSERT_FALSE``, reducing the
  number of false positives in test code.

- Improved :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` check by adding the function
  ``std::get_temporary_buffer`` to the default list of unsafe functions. (This
  function is unsafe, useless, deprecated in C++17 and removed in C++20).

- Improved :doc:`bugprone-use-after-move
  <clang-tidy/checks/bugprone/use-after-move>` check:

  - Include the name of the invalidating function in the warning message when a
    custom invalidation function is used (via the `InvalidationFunctions`
    option).

  - Add support for annotation of user-defined types as having the same
    moved-from semantics as standard smart pointers.

  - Do not report explicit call to destructor after move as an invalid use.

- Improved :doc:`cppcoreguidelines-avoid-capturing-lambda-coroutines
  <clang-tidy/checks/cppcoreguidelines/avoid-capturing-lambda-coroutines>`
  check by adding the `AllowExplicitObjectParameters` option. When enabled,
  lambda coroutines using C++23 deducing ``this`` (explicit object parameter)
  are not flagged.

- Improved :doc:`cppcoreguidelines-init-variables
  <clang-tidy/checks/cppcoreguidelines/init-variables>` check by ensuring that
  member pointers are correctly flagged as uninitialized.

- Improved :doc:`cppcoreguidelines-missing-std-forward
  <clang-tidy/checks/cppcoreguidelines/missing-std-forward>` check:
  
  - Fixed false positive for constrained template parameters
  
  - Fixed false positive with ``std::forward`` in brace-init and paren-init
    lambda captures such as ``[t{std::forward<T>(t)}]``.

- Improved :doc:`cppcoreguidelines-pro-type-member-init
  <clang-tidy/checks/cppcoreguidelines/pro-type-member-init>` check by fixing
  a false positive when a base class has a forward declaration before its
  definition.

- Improved :doc:`cppcoreguidelines-pro-type-vararg
  <clang-tidy/checks/cppcoreguidelines/pro-type-vararg>` check by no longer
  warning on builtins with custom type checking (e.g., type-generic builtins
  like ``__builtin_clzg``) that use variadic declarations as an implementation
  detail.

- Improved :doc:`cppcoreguidelines-rvalue-reference-param-not-moved
  <clang-tidy/checks/cppcoreguidelines/rvalue-reference-param-not-moved>` check
  by fixing a false positive on implicitly generated functions such as
  inherited constructors.

- Improved :doc:`llvm-use-ranges
  <clang-tidy/checks/llvm/use-ranges>` check by adding support for the following
  algorithms: ``std::accumulate``, ``std::replace_copy``, and
  ``std::replace_copy_if``.

- Improved :doc:`misc-const-correctness
  <clang-tidy/checks/misc/const-correctness>` check:

  - Added support for analyzing function parameters with the `AnalyzeParameters`
    option.

  - Fixed false positive where an array of pointers to ``const`` was
    incorrectly diagnosed as allowing the pointee to be made ``const``.

  - Fixed false positive where a pointer used with placement new was
    incorrectly diagnosed as allowing the pointee to be made ``const``.

- Improved :doc:`misc-multiple-inheritance
  <clang-tidy/checks/misc/multiple-inheritance>` by avoiding false positives when
  virtual inheritance causes concrete bases to be counted more than once.

- Improved :doc:`misc-throw-by-value-catch-by-reference
  <clang-tidy/checks/misc/throw-by-value-catch-by-reference>` check:

  - Fixed the `WarnOnLargeObject` option to use the correct name when
    storing the configuration.

  - Fixed the `CheckThrowTemporaries` option to correctly reflect its
    configured value in exported settings.

- Improved :doc:`misc-unused-using-decls
  <clang-tidy/checks/misc/unused-using-decls>` to not diagnose ``using``
  declarations as unused if they're exported from a module.

- Improved :doc:`misc-use-internal-linkage
  <clang-tidy/checks/misc/use-internal-linkage>` to not suggest giving
  internal linkage to entities defined in C++ module interface units.
  Because it only sees one file at a time, the check can't be sure
  such entities aren't referenced in any other files of that module.

- Improved :doc:`modernize-pass-by-value
  <clang-tidy/checks/modernize/pass-by-value>` check by adding `IgnoreMacros`
  option to suppress warnings in macros.

- Improved :doc:`modernize-redundant-void-arg
  <clang-tidy/checks/modernize/redundant-void-arg>` check to work in C23.

- Improved :doc:`modernize-use-equals-delete
  <clang-tidy/checks/modernize/use-equals-delete>` check by only warning on
  private deleted functions, if they do not have a public overload or are a
  special member function.

- Improved :doc:`modernize-use-std-format
  <clang-tidy/checks/modernize/use-std-format>` check:

  - Fixed a crash when an argument is part of a macro expansion.

  - Added missing ``#include`` insertion when the format function call
    appears as an argument to a macro.

- Improved :doc:`modernize-use-std-print
  <clang-tidy/checks/modernize/use-std-print>` check by adding missing
  ``#include`` insertion when the format function call appears as an
  argument to a macro.

- Improved :doc:`modernize-use-trailing-return-type
  <clang-tidy/checks/modernize/use-trailing-return-type>` check by fixing
  spurious ``missing '(' after '__has_feature'`` errors caused by builtin
  macros appearing in the return type of a function.

- Improved :doc:`modernize-use-using
  <clang-tidy/checks/modernize/use-using>` check by avoiding the generation
  of invalid code for function types with redundant parentheses.

- Improved :doc:`performance-enum-size
  <clang-tidy/checks/performance/enum-size>` check:

  - Exclude ``enum`` in ``extern "C"`` blocks.

  - Improved the ignore list to correctly handle ``typedef`` and  ``enum``.

- Improved :doc:`performance-inefficient-string-concatenation
  <clang-tidy/checks/performance/inefficient-string-concatenation>` check by
  adding support for detecting inefficient string concatenation in ``do-while``
  loops.

- Improved :doc:`performance-inefficient-vector-operation
  <clang-tidy/checks/performance/inefficient-vector-operation>` check by
  correctly handling vector-like classes when ``push_back``/``emplace_back`` are
  inherited.

- Improved :doc:`performance-move-const-arg
  <clang-tidy/checks/performance/move-const-arg>` check by avoiding false
  positives on trivially copyable types with a non-public copy constructor.

- Improved :doc:`performance-prefer-single-char-overloads
  <clang-tidy/checks/performance/prefer-single-char-overloads>` check:

  - Now analyzes calls to the ``starts_with``, ``ends_with``, ``contains``,
    and ``operator+=`` string member functions.

  - Fixes false negatives when using ``std::set`` from ``libstdc++``.

- Improved :doc:`performance-trivially-destructible
  <clang-tidy/checks/performance/trivially-destructible>` check by fixing
  false positives when a class is seen through both a header include and
  a C++20 module import.

- Improved :doc:`readability-container-size-empty
  <clang-tidy/checks/readability/container-size-empty>` check:

  - Fix a crash when a member expression has a non-identifier name.

  - Reduce verbosity by removing the note indicating source location of the
    ``empty`` function.

- Improved :doc:`readability-else-after-return
  <clang-tidy/checks/readability/else-after-return>` check:

  - Fixed missed diagnostics when ``if`` statements appear in unbraced
    ``switch`` case labels.

  - Fixed a false positive involving ``if`` statements which contain
    a ``return``, ``break``, etc., jumped over by a ``goto``.

  - Fixed the check potentially breaking code by deleting one too many
    characters following an ``else`` or a curly brace.

  - Added support for handling attributed ``if`` then-branches such as
    ``[[likely]]`` and ``[[unlikely]]``.

  - Diagnose and remove redundant ``else`` branches after calls to
    ``[[noreturn]]`` functions.

- Improved :doc:`readability-enum-initial-value
  <clang-tidy/checks/readability/enum-initial-value>` check: the warning message
  now uses separate note diagnostics for each uninitialized enumerator, making
  it easier to see which specific enumerators need explicit initialization.

- Improved :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check:

  - Fixed incorrect naming style application to C++17 structured bindings.

  - Fixed a false positive where function templates could be diagnosed as generic 
    identifiers when `DefaultCase` was enabled.

- Improved :doc:`readability-implicit-bool-conversion
  <clang-tidy/checks/readability/implicit-bool-conversion>` check:

  - Fixed a false positive where `AllowPointerConditions` and
    `AllowIntegerConditions` options did not suppress warnings when the
    condition expression involved temporaries (e.g. passing a string literal to
    a ``const std::string&`` parameter).

  - Warn and provide fix-its when a macro defined in a system header (e.g.
    ``NULL``) is implicitly converted to ``bool``.

  - Added `AllowLogicalOperatorConversion` option to suppress warnings on
    implicit conversions of logical operator results (``&&``, ``||``, ``!``)
    to ``bool`` in C.

- Improved :doc:`readability-non-const-parameter
  <clang-tidy/checks/readability/non-const-parameter>` check:

  - Avoid false positives on parameters used in dependent expressions
    (e.g. inside generic lambdas).

  - Fixed a false positive in array subscript expressions where the types are
    not yet resolved.

- Improved :doc:`readability-redundant-preprocessor
  <clang-tidy/checks/readability/redundant-preprocessor>` check by fixing a
  false positive for nested ``#if`` directives using different builtin
  expressions such as ``__has_builtin`` and ``__has_cpp_attribute``.

- Improved :doc:`readability-simplify-boolean-expr
  <clang-tidy/checks/readability/simplify-boolean-expr>` check to provide valid
  fix suggestions for C23 and later by not using ``static_cast``.

- Improved :doc:`readability-simplify-subscript-expr
  <clang-tidy/checks/readability/simplify-subscript-expr>` check by fixing
  missing warnings when subscripting an object held inside a generic
  container (e.g. subscripting a ``std::string`` held inside a
  ``std::vector<std::string>``).

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
