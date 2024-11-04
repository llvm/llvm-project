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

- The tweak for turning unscoped into scoped enums now removes redundant prefixes
  from the enum values.

Signature help
^^^^^^^^^^^^^^

Cross-references
^^^^^^^^^^^^^^^^

Objective-C
^^^^^^^^^^^

Miscellaneous
^^^^^^^^^^^^^

- Added a boolean option `AnalyzeAngledIncludes` to `Includes` config section,
  which allows to enable unused includes detection for all angled ("system") headers.
  At this moment umbrella headers are not supported, so enabling this option
  may result in false-positives.

Improvements to clang-doc
-------------------------

Improvements to clang-query
---------------------------

- Added the `file` command to dynamically load a list of commands and matchers
  from an external file, allowing the cost of reading the compilation database
  and building the AST to be imposed just once for faster prototyping.

- Removed support for ``enable output srcloc``. Fixes #GH82591

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- Improved :program:`run-clang-tidy.py` script. Added argument `-source-filter`
  to filter source files from the compilation database, via a RegEx. In a
  similar fashion to what `-header-filter` does for header files.

- Improved :program:`check_clang_tidy.py` script. Added argument `-export-fixes`
  to aid in clang-tidy and test development.

- Fixed bug where big values for unsigned check options overflowed into negative values
  when being printed with `--dump-config`.

- Fixed `--verify-config` option not properly parsing checks when using the
  literal operator in the `.clang-tidy` config.

- Added argument `--exclude-header-filter` and config option `ExcludeHeaderFilterRegex`
  to exclude headers from analysis via a RegEx.

New checks
^^^^^^^^^^

- New :doc:`bugprone-crtp-constructor-accessibility
  <clang-tidy/checks/bugprone/crtp-constructor-accessibility>` check.

  Detects error-prone Curiously Recurring Template Pattern usage, when the CRTP
  can be constructed outside itself and the derived class.

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

- New :doc:`modernize-use-designated-initializers
  <clang-tidy/checks/modernize/use-designated-initializers>` check.

  Finds initializer lists for aggregate types that could be
  written as designated initializers instead.

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

- New alias :doc:`cert-int09-c <clang-tidy/checks/cert/int09-c>` to
  :doc:`readability-enum-initial-value <clang-tidy/checks/readability/enum-initial-value>`
  was added.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-assert-side-effect
  <clang-tidy/checks/bugprone/assert-side-effect>` check by detecting side
  effect from calling a method with non-const reference parameters.

- Improved :doc:`bugprone-casting-through-void
  <clang-tidy/checks/bugprone/casting-through-void>` check by ignoring casts
  where source is already a ``void``` pointer, making middle ``void`` pointer
  casts bug-free.

- Improved :doc:`bugprone-forwarding-reference-overload
  <clang-tidy/checks/bugprone/forwarding-reference-overload>`
  check to ignore deleted constructors which won't hide other overloads.

- Improved :doc:`bugprone-inc-dec-in-conditions
  <clang-tidy/checks/bugprone/inc-dec-in-conditions>` check to ignore code
  within unevaluated contexts, such as ``decltype``.

- Improved :doc:`bugprone-lambda-function-name<clang-tidy/checks/bugprone/lambda-function-name>`
  check by ignoring ``__func__`` macro in lambda captures, initializers of
  default parameters and nested function declarations.

- Improved :doc:`bugprone-multi-level-implicit-pointer-conversion
  <clang-tidy/checks/bugprone/multi-level-implicit-pointer-conversion>` check
  by ignoring implicit pointer conversions that are part of a cast expression.

- Improved :doc:`bugprone-non-zero-enum-to-bool-conversion
  <clang-tidy/checks/bugprone/non-zero-enum-to-bool-conversion>` check by
  eliminating false positives resulting from direct usage of bitwise operators
  within parentheses.

- Improved :doc:`bugprone-optional-value-conversion
  <clang-tidy/checks/bugprone/optional-value-conversion>` check by eliminating
  false positives resulting from use of optionals in unevaluated context.

- Improved :doc:`bugprone-sizeof-expression
  <clang-tidy/checks/bugprone/sizeof-expression>` check by eliminating some
  false positives and adding a new (off-by-default) option
  `WarnOnSizeOfPointer` that reports all ``sizeof(pointer)`` expressions
  (except for a few that are idiomatic).

- Improved :doc:`bugprone-suspicious-include
  <clang-tidy/checks/bugprone/suspicious-include>` check by replacing the local
  options `HeaderFileExtensions` and `ImplementationFileExtensions` by the
  global options of the same name.

- Improved :doc:`bugprone-too-small-loop-variable
  <clang-tidy/checks/bugprone/too-small-loop-variable>` check by incorporating
  better support for ``const`` loop boundaries.

- Improved :doc:`bugprone-unused-local-non-trivial-variable
  <clang-tidy/checks/bugprone/unused-local-non-trivial-variable>` check by
  ignoring local variable with ``[maybe_unused]`` attribute.

- Improved :doc:`bugprone-unused-return-value
  <clang-tidy/checks/bugprone/unused-return-value>` check by updating the
  parameter `CheckedFunctions` to support regexp, avoiding false positive for
  function with the same prefix as the default argument, e.g. ``std::unique_ptr``
  and ``std::unique``, avoiding false positive for assignment operator overloading.

- Improved :doc:`bugprone-use-after-move
  <clang-tidy/checks/bugprone/use-after-move>` check to also handle
  calls to ``std::forward``.

- Improved :doc:`cppcoreguidelines-macro-usage
  <clang-tidy/checks/cppcoreguidelines/macro-usage>` check by ignoring macro with
  hash preprocessing token.

- Improved :doc:`cppcoreguidelines-missing-std-forward
  <clang-tidy/checks/cppcoreguidelines/missing-std-forward>` check by no longer
  giving false positives for deleted functions, by fixing false negatives when only
  a few parameters are forwarded and by ignoring parameters without a name (unused
  arguments).

- Improved :doc:`cppcoreguidelines-owning-memory
  <clang-tidy/checks/cppcoreguidelines/owning-memory>` check to properly handle
  return type in lambdas and in nested functions.

- Improved :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines/prefer-member-initializer>` check
  by removing enforcement of rule `C.48
  <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c48-prefer-in-class-initializers-to-member-initializers-in-constructors-for-constant-initializers>`_,
  which was deprecated since :program:`clang-tidy` 17. This rule is now covered
  by :doc:`cppcoreguidelines-use-default-member-init
  <clang-tidy/checks/cppcoreguidelines/use-default-member-init>`. Fixed
  incorrect hints when using list-initialization.

- Improved :doc:`cppcoreguidelines-special-member-functions
  <clang-tidy/checks/cppcoreguidelines/special-member-functions>` check with a
  new option `AllowImplicitlyDeletedCopyOrMove`, which removes the requirement
  for explicit copy or move special member functions when they are already
  implicitly deleted.

- Improved :doc:`google-build-namespaces
  <clang-tidy/checks/google/build-namespaces>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.

- Improved :doc:`google-explicit-constructor
  <clang-tidy/checks/google/explicit-constructor>` check to better handle
  C++20 `explicit(bool)`.

- Improved :doc:`google-global-names-in-headers
  <clang-tidy/checks/google/global-names-in-headers>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.

- Improved :doc:`google-runtime-int <clang-tidy/checks/google/runtime-int>`
  check performance through optimizations.

- Improved :doc:`hicpp-signed-bitwise <clang-tidy/checks/hicpp/signed-bitwise>`
  check by ignoring false positives involving positive integer literals behind
  implicit casts when `IgnorePositiveIntegerLiterals` is enabled.

- Improved :doc:`hicpp-ignored-remove-result <clang-tidy/checks/hicpp/ignored-remove-result>`
  check by ignoring other functions with same prefixes as the target specific
  functions.

- Improved :doc:`linuxkernel-must-check-errs
  <clang-tidy/checks/linuxkernel/must-check-errs>` check documentation to
  consistently use the check's proper name.

- Improved :doc:`llvm-header-guard
  <clang-tidy/checks/llvm/header-guard>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.

- Improved :doc:`misc-const-correctness
  <clang-tidy/checks/misc/const-correctness>` check by avoiding infinite recursion
  for recursive functions with forwarding reference parameters and reference
  variables which refer to themselves.

- Improved :doc:`misc-definitions-in-headers
  <clang-tidy/checks/misc/definitions-in-headers>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.
  Additionally, the option `UseHeaderFileExtensions` is removed, so that the
  check uses the `HeaderFileExtensions` option unconditionally.

- Improved :doc:`misc-header-include-cycle
  <clang-tidy/checks/misc/header-include-cycle>` check by avoiding crash for self
  include cycles.

- Improved :doc:`misc-unused-using-decls
  <clang-tidy/checks/misc/unused-using-decls>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.

- Improved :doc:`misc-use-anonymous-namespace
  <clang-tidy/checks/misc/use-anonymous-namespace>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.

- Improved :doc:`modernize-avoid-c-arrays
  <clang-tidy/checks/modernize/avoid-c-arrays>` check by introducing the new
  `AllowStringArrays` option, enabling the exclusion of array types with deduced
  length initialized from string literals.

- Improved :doc:`modernize-loop-convert
  <clang-tidy/checks/modernize/loop-convert>` check by ensuring that fix-its
  don't remove parentheses used in ``sizeof`` calls when they have array index
  accesses as arguments.

- Improved :doc:`modernize-use-constraints
  <clang-tidy/checks/modernize/use-constraints>` check by fixing a crash that
  occurred in some scenarios and excluding system headers from analysis.

- Improved :doc:`modernize-use-nullptr
  <clang-tidy/checks/modernize/use-nullptr>` check to include support for C23,
  which also has introduced the ``nullptr`` keyword.

- Improved :doc:`modernize-use-override
  <clang-tidy/checks/modernize/use-override>` check to also remove any trailing
  whitespace when deleting the ``virtual`` keyword.

- Improved :doc:`modernize-use-starts-ends-with
  <clang-tidy/checks/modernize/use-starts-ends-with>` check to also handle
  calls to ``compare`` method.

- Improved :doc:`modernize-use-std-print
  <clang-tidy/checks/modernize/use-std-print>` check to not crash if the
  format string parameter of the function to be replaced is not of the
  expected type.

- Improved :doc:`modernize-use-using <clang-tidy/checks/modernize/use-using>`
  check by adding support for detection of typedefs declared on function level.

- Improved :doc:`performance-inefficient-vector-operation
  <clang-tidy/checks/performance/inefficient-vector-operation>` fixing false
  negatives caused by different variable definition type and variable initial
  value type in loop initialization expression.

- Improved :doc:`performance-move-const-arg
  <clang-tidy/checks/performance/move-const-arg>` check by ignoring
  ``std::move()`` calls when their target is used as an rvalue.

- Improved :doc:`performance-unnecessary-copy-initialization
  <clang-tidy/checks/performance/unnecessary-copy-initialization>` check by
  detecting more cases of constant access. In particular, pointers can be
  analyzed, so the check now handles the common patterns
  `const auto e = (*vector_ptr)[i]` and `const auto e = vector_ptr->at(i);`.
  Calls to mutable function where there exists a `const` overload are also
  handled.

- Improved :doc:`readability-avoid-return-with-void-value
  <clang-tidy/checks/readability/avoid-return-with-void-value>` check by adding
  fix-its.

- Improved :doc:`readability-const-return-type
  <clang-tidy/checks/readability/const-return-type>` check to eliminate false
  positives when returning types with const not at the top level.

- Improved :doc:`readability-container-size-empty
  <clang-tidy/checks/readability/container-size-empty>` check to prevent false
  positives when utilizing ``size`` or ``length`` methods that accept parameter.
  Fixed crash when facing template user defined literals.

- Improved :doc:`readability-duplicate-include
  <clang-tidy/checks/readability/duplicate-include>` check by excluding include
  directives that form the filename using macro.

- Improved :doc:`readability-else-after-return
  <clang-tidy/checks/readability/else-after-return>` check to ignore
  `if consteval` statements, for which the `else` branch must not be removed.

- Improved :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check in `GetConfigPerFile`
  mode by resolving symbolic links to header files. Fixed handling of Hungarian
  Prefix when configured to `LowerCase`. Added support for renaming designated
  initializers. Added support for renaming macro arguments. Fixed renaming
  conflicts arising from out-of-line member function template definitions.

- Improved :doc:`readability-implicit-bool-conversion
  <clang-tidy/checks/readability/implicit-bool-conversion>` check to provide
  valid fix suggestions for ``static_cast`` without a preceding space and
  fixed problem with duplicate parentheses in double implicit casts. Corrected
  the fix suggestions for C23 and later by using C-style casts instead of
  ``static_cast``. Fixed false positives in C++20 spaceship operator by ignoring
  casts in implicit and defaulted functions.

- Improved :doc:`readability-redundant-inline-specifier
  <clang-tidy/checks/readability/redundant-inline-specifier>` check to properly
  emit warnings for static data member with an in-class initializer.

- Improved :doc:`readability-redundant-member-init
  <clang-tidy/checks/readability/redundant-member-init>` check to avoid
  false-positives when type of the member does not match the type of the
  initializer.

- Improved :doc:`readability-static-accessed-through-instance
  <clang-tidy/checks/readability/static-accessed-through-instance>` check to
  support calls to overloaded operators as base expression and provide fixes to
  expressions with side-effects.

- Improved :doc:`readability-simplify-boolean-expr
  <clang-tidy/checks/readability/simplify-boolean-expr>` check to avoid to emit
  warning for macro when IgnoreMacro option is enabled.

- Improved :doc:`readability-static-definition-in-anonymous-namespace
  <clang-tidy/checks/readability/static-definition-in-anonymous-namespace>`
  check by resolving fix-it overlaps in template code by disregarding implicit
  instances.

- Improved :doc:`readability-string-compare
  <clang-tidy/checks/readability/string-compare>` check to also detect
  usages of ``std::string_view::compare``. Added a `StringLikeClasses` option
  to detect usages of ``compare`` method in custom string-like classes.

Removed checks
^^^^^^^^^^^^^^

- Removed `cert-dcl21-cpp`, which was deprecated since :program:`clang-tidy` 17,
  since the rule DCL21-CPP has been removed from the CERT guidelines.

Miscellaneous
^^^^^^^^^^^^^

- Fixed incorrect formatting in :program:`clang-apply-replacements` when no
  `--format` option is specified. Now :program:`clang-apply-replacements`
  applies formatting only with the option.

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
