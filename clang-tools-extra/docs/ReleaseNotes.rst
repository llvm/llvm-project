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
  <clang-tidy/checks/misc/const-correctness>` check to avoid false
  positives when pointers is transferred to non-const references
  and avoid false positives of function pointer and fix false
  positives on return of non-const pointer and fix false positives on
  pointer-to-member operator and avoid false positives when the address
  of a variable is taken to be passed to a function.

- Improved :doc:`misc-coroutine-hostile-raii
  <clang-tidy/checks/misc/coroutine-hostile-raii>` check by adding the option
  `AllowedCallees`, that allows exempting safely awaitable callees from the
  check.

- Improved :doc:`misc-header-include-cycle
  <clang-tidy/checks/misc/header-include-cycle>` check performance.

- Improved :doc:`misc-use-internal-linkage
  <clang-tidy/checks/misc/use-internal-linkage>` to suggest giving
  user-defined types (structs, classes, unions, and enums) internal
  linkage. Added fine-grained options to control whether the check
  should diagnose functions, variables, and/or user-defined types.
  Enabled the check for C.

- Improved :doc:`modernize-avoid-c-arrays
  <clang-tidy/checks/modernize/avoid-c-arrays>` to not diagnose array types
  which are part of an implicit instantiation of a template.

- Improved :doc:`modernize-use-constraints
  <clang-tidy/checks/modernize/use-constraints>` check by fixing a crash on
  uses of non-standard ``enable_if`` with a signature different from
  ``std::enable_if`` (such as ``boost::enable_if``).

- Improved :doc:`modernize-use-default-member-init
  <clang-tidy/checks/modernize/use-default-member-init>` check to
  enhance the robustness of the member initializer detection.

- Improved :doc:`modernize-use-designated-initializers
  <clang-tidy/checks/modernize/use-designated-initializers>` check to
  suggest using designated initializers for aliased aggregate types.

- Improved :doc:`modernize-use-integer-sign-comparison
  <clang-tidy/checks/modernize/use-integer-sign-comparison>` by providing
  correct fix-its when the right-hand side of a comparison contains a
  non-C-style cast.

- Improved :doc:`modernize-use-nullptr
  <clang-tidy/checks/modernize/use-nullptr>` check by fixing a crash
  on Windows when the check was enabled with a 32-bit :program:`clang-tidy`
  binary.

- Improved :doc:`modernize-use-override
  <clang-tidy/checks/modernize/use-override>` by fixing an issue where
  the check would sometimes suggest inserting ``override`` in an invalid
  place.

- Improved :doc:`modernize-use-ranges
  <clang-tidy/checks/modernize/use-ranges>` check to suggest using
  the more idiomatic ``std::views::reverse`` where it used to suggest
  ``std::ranges::reverse_view``.

- Improved :doc:`modernize-use-scoped-lock
  <clang-tidy/checks/modernize/use-scoped-lock>` check by fixing a crash
  on malformed code (common when using :program:`clang-tidy` through
  :program:`clangd`).

- Improved :doc:`modernize-use-std-format
  <clang-tidy/checks/modernize/use-std-format>` check to correctly match
  when the format string is converted to a different type by an implicit
  constructor call.

- Improved :doc:`modernize-use-std-print
  <clang-tidy/checks/modernize/use-std-print>` check to correctly match
  when the format string is converted to a different type by an implicit
  constructor call, and fixed a crash when handling format strings
  containing non-ASCII characters.

- Improved :doc:`modernize-use-using
  <clang-tidy/checks/modernize/use-using>` check to correctly provide fix-its
  for typedefs of pointers or references to array types.

- Improved :doc:`performance-unnecessary-copy-initialization
  <clang-tidy/checks/performance/unnecessary-copy-initialization>` by printing
  the type of the diagnosed variable.

- Improved :doc:`performance-unnecessary-value-param
  <clang-tidy/checks/performance/unnecessary-value-param>` by printing
  the type of the diagnosed variable and correctly generating fix-it hints for
  parameter-pack arguments.

- Improved :doc:`portability-template-virtual-member-function
  <clang-tidy/checks/portability/template-virtual-member-function>` check to
  avoid false positives on pure virtual member functions.

- Improved :doc:`readability-container-contains
  <clang-tidy/checks/readability/container-contains>` to support string
  comparisons to ``npos``. Internal changes may cause new rare false positives
  in non-standard containers.

- Improved :doc:`readability-container-data-pointer
  <clang-tidy/checks/readability/container-data-pointer>` check by correctly
  adding parentheses when the container expression is a dereference.

- Improved :doc:`readability-container-size-empty
  <clang-tidy/checks/readability/container-size-empty>` check by correctly
  generating fix-it hints when size method is called from implicit ``this``,
  ignoring default constructors with user provided arguments and adding
  detection in container's method except ``empty``.

- Improved :doc:`readability-duplicate-include
  <clang-tidy/checks/readability/duplicate-include>` check by adding
  the ``IgnoredFilesList`` option (semicolon-separated list of regexes or
  filenames) to allow intentional duplicates.

- Improved :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check by ignoring
  declarations and macros in system headers. The documentation is also improved
  to differentiate the general options from the specific ones. Options for
  fine-grained control over ``constexpr`` variables were added. Added default
  options which simplify configs by removing the need to specify each
  identifier kind separately.

- Improved :doc:`readability-implicit-bool-conversion
  <clang-tidy/checks/readability/implicit-bool-conversion>` check by correctly
  adding parentheses when inner expressions are implicitly converted multiple
  times, enabling the check for C99 and later standards, and allowing implicit
  conversions from ``bool`` to integer when used as operands of logical
  operators (``&&``, ``||``) in C.

- Improved :doc:`readability-inconsistent-declaration-parameter-name
  <clang-tidy/checks/readability/inconsistent-declaration-parameter-name>` check
  by not enforcing parameter name consistency between a variadic parameter pack
  in the primary template and specific parameters in its specializations.

- Improved :doc:`readability-make-member-function-const
  <clang-tidy/checks/readability/make-member-function-const>` check by fixing
  false positives when accessing pointer or reference members inside unions.

- Improved :doc:`readability-math-missing-parentheses
  <clang-tidy/checks/readability/math-missing-parentheses>` check by correctly
  diagnosing operator precedence issues inside parenthesized expressions.

- Improved :doc:`readability-qualified-auto
  <clang-tidy/checks/readability/qualified-auto>` check by adding the option
  `IgnoreAliasing`, that allows not looking at underlying types of type aliases.

- Improved :doc:`readability-redundant-casting
  <clang-tidy/checks/readability/redundant-casting>` check by fixing false
  negatives when explicitly cast from function pointer.

- Improved :doc:`readability-redundant-control-flow
  <clang-tidy/checks/readability/redundant-control-flow>` by fixing an issue
  where the check would sometimes suggest deleting not only a redundant
  ``return`` or ``continue``, but also unrelated lines preceding it.

- Improved :doc:`readability-uppercase-literal-suffix
  <clang-tidy/checks/readability/uppercase-literal-suffix>` check to recognize
  literal suffixes added in C++23 and C23.

- Improved :doc:`readability-use-concise-preprocessor-directives
  <clang-tidy/checks/readability/use-concise-preprocessor-directives>` check to
  generate correct fix-its for forms without a space after the directive.

- Improved :doc:`readability-use-std-min-max
  <clang-tidy/checks/readability/use-std-min-max>` check by ensuring that
  comments between the ``if`` condition and the ``then`` block are preserved
  when applying the fix.
  <clang-tidy/checks/misc/const-correctness>` check:

  - Added support for analyzing function parameters with the `AnalyzeParameters`
    option.

  - Fixed false positive where an array of pointers to ``const`` was
    incorrectly diagnosed as allowing the pointee to be made ``const``.

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
