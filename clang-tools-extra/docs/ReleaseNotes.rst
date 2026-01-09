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

- Deprecated the :program:`clang-tidy` ``zircon`` module. All checks have been
  moved to the ``fuchsia`` module instead. The ``zircon`` module will be removed
  in the 24th release.

- Removed :program:`clang-tidy`'s global options `IgnoreMacros` and
  `StrictMode`, which were documented as deprecated since
  :program:`clang-tidy-20`. Users should use the check-specific options of the
  same name instead.

- Removed `clang-analyzer-*` checks from default checks in :program:`clang-tidy`.
  From now on, users should specify explicitly that they want CSA checks to run
  in :program:`clang-tidy` via `clang-analyzer-*`.

- Renamed a few :program:`clang-tidy` check options, as they
  were misspelled:

  - `NamePrefixSuffixSilenceDissimilarityTreshold` to
    `NamePrefixSuffixSilenceDissimilarityThreshold` in
    :doc:`bugprone-easily-swappable-parameters
    <clang-tidy/checks/bugprone/easily-swappable-parameters>`

  - `CharTypdefsToIgnore` to `CharTypedefsToIgnore` in
    :doc:`bugprone-signed-char-misuse
    <clang-tidy/checks/bugprone/signed-char-misuse>`

- Modified the custom message format of :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` by assigning a special meaning
  to the character ``>`` at the start of the value of the option
  ``CustomFunctions``. If the option value starts with ``>``, then the
  replacement suggestion part of the message (which would be included by
  default) is omitted. (This does not change the warning locations.)

- :program:`clang-tidy` now displays warnings from all non-system headers by
  default. Previously, users had to explicitly opt-in to header warnings using
  `-header-filter='.*'`. To disable warnings from non-system, set `-header-filter`
  to an empty string.

- The ``clang-tidy/ClangTidyModuleRegistry.h`` header has been deprecated and will
  be removed in LLVM 24. All of the symbols it used to define have been moved into
  ``clang-tidy/ClangTidyModule.h``.

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

- Added a new ``MacroFilter`` configuration option to ``Completion`` to
  allow fuzzy-matching with the ``FuzzyMatch`` option when suggesting
  macros. ``ExactPrefix`` is the default, which retains previous
  behavior of suggesting macros which match the prefix exactly.

Code actions
^^^^^^^^^^^^

- New ``Override pure virtual methods`` code action. When invoked on a class
  definition, this action automatically generates C++ ``override`` declarations
  for all pure virtual methods inherited from its base classes that have not yet
  been implemented. The generated method stubs prompts the user for the actual
  implementation. The overrides are intelligently grouped under their original
  access specifiers (e.g., ``public``, ``protected``), creating new access
  specifier blocks if necessary.

Signature help
^^^^^^^^^^^^^^

Cross-references
^^^^^^^^^^^^^^^^

Objective-C
^^^^^^^^^^^

Miscellaneous
^^^^^^^^^^^^^

- Add wildcard ``.gitignore`` file to the clangd index directory.

Improvements to clang-doc
-------------------------

Improvements to clang-query
---------------------------

- Matcher queries interpreted by clang-query are now support trailing comma (,)
  in matcher arguments. Note that C++ still doesn't allow this in function
  arguments. So when porting a query to C++, remove all instances of trailing
  comma (otherwise C++ compiler will just complain about "expected expression").

Improvements to clang-tidy
--------------------------

- The :program:`check_clang_tidy.py` tool now recognizes the ``-std`` argument
  when run over C files. If ``-std`` is not specified, it defaults to
  ``c99-or-later``.

- :program:`clang-tidy` now displays warnings from all non-system headers by
  default. Previously, users had to explicitly opt-in to header warnings using
  `-header-filter='.*'`. To disable warnings from non-system, set `-header-filter`
  to an empty string.

- :program:`clang-tidy` no longer attempts to analyze code from system headers
  by default, greatly improving performance. This behavior is disabled if the
  `SystemHeaders` option is enabled.

- :program:`clang-tidy` now supports query based custom checks by `CustomChecks`
  configuration option.
  :doc:`Query Based Custom Check Document <clang-tidy/QueryBasedCustomChecks>`

- The :program:`run-clang-tidy.py` and :program:`clang-tidy-diff.py` scripts
  now run checks in parallel by default using all available hardware threads.
  Both scripts display the number of threads being used in their output.

- Improved :program:`run-clang-tidy.py` by adding a new option
  `enable-check-profile` to enable per-check timing profiles and print a
  report based on all analyzed files.

- Improved documentation of the `-line-filter` command-line flag of
  :program:`clang-tidy` and :program:`run-clang-tidy.py`.

- Improved :program:`clang-tidy` option `-quiet` by suppressing diagnostic
  count messages.

- Improved :program:`clang-tidy` by not crashing when an empty `directory`
  field is used in a compilation database; the current working directory
  will be used instead, and an error message will be printed.

- Removed :program:`clang-tidy`'s global options `IgnoreMacros` and
  `StrictMode`, which were documented as deprecated since
  :program:`clang-tidy-20`. Users should use the check-specific options of the
  same name instead.

- Improved :program:`run-clang-tidy.py` and :program:`clang-tidy-diff.py`
  scripts by adding the `-hide-progress` option to suppress progress and
  informational messages.

- Removed `clang-analyzer-*` check from default checks in :program:`clang-tidy`.
  From now on, users should specify explicitly that they want CSA checks to run
  in :program:`clang-tidy`.

- Improved :program:`clang-tidy` by adding the `--removed-arg` option to remove
  arguments sent to the compiler when invoking Clang-Tidy. This option was also
  added to :program:`run-clang-tidy.py` and :program:`clang-tidy-diff.py` and
  can be configured in the config file through the `RemovedArgs` option.

- Deprecated the :program:`clang-tidy` ``zircon`` module. All checks have been
  moved to the ``fuchsia`` module instead. The ``zircon`` module will be removed
  in the 24th release.

- Improved :program:`clang-tidy` configuration parsing by allowing the same list
  syntax in `WarningsAsErrors` as in `Checks`.

New checks
^^^^^^^^^^

- New :doc:`abseil-unchecked-statusor-access
  <clang-tidy/checks/abseil/unchecked-statusor-access>` check.

  Finds uses of ``absl::StatusOr`` without checking if a value is present.

- New :doc:`bugprone-derived-method-shadowing-base-method
  <clang-tidy/checks/bugprone/derived-method-shadowing-base-method>` check.

  Finds derived class methods that shadow a (non-virtual) base class method.

- New :doc:`bugprone-invalid-enum-default-initialization
  <clang-tidy/checks/bugprone/invalid-enum-default-initialization>` check.

  Detects default initialization (to 0) of variables with ``enum`` type where
  the enum has no enumerator with value of 0.

- New :doc:`cppcoreguidelines-pro-bounds-avoid-unchecked-container-access
  <clang-tidy/checks/cppcoreguidelines/pro-bounds-avoid-unchecked-container-access>`
  check.

  Finds calls to ``operator[]`` in STL containers and suggests replacing them
  with safe alternatives.

- New :doc:`google-runtime-float
  <clang-tidy/checks/google/runtime-float>` check.

  Finds uses of ``long double`` and suggests against their use due to lack of
  portability.

- New :doc:`llvm-mlir-op-builder
  <clang-tidy/checks/llvm/use-new-mlir-op-builder>` check.

  Checks for uses of MLIR's old/to be deprecated ``OpBuilder::create<T>`` form
  and suggests using ``T::create`` instead.

- New :doc:`llvm-use-ranges
  <clang-tidy/checks/llvm/use-ranges>` check.

  Finds calls to STL library iterator algorithms that could be replaced with
  LLVM range-based algorithms from ``llvm/ADT/STLExtras.h``.

- New :doc:`misc-override-with-different-visibility
  <clang-tidy/checks/misc/override-with-different-visibility>` check.

  Finds virtual function overrides with different visibility than the function
  in the base class.

- New :doc:`readability-inconsistent-ifelse-braces
  <clang-tidy/checks/readability/inconsistent-ifelse-braces>` check.

  Detects ``if``/``else`` statements where one branch uses braces and the other
  does not.

- New :doc:`readability-redundant-parentheses
  <clang-tidy/checks/readability/redundant-parentheses>` check.

  Detect redundant parentheses.

- New :doc:`readability-redundant-typename
  <clang-tidy/checks/readability/redundant-typename>` check.

  Finds redundant uses of the ``typename`` keyword.

New check aliases
^^^^^^^^^^^^^^^^^

- Renamed :doc:`cert-dcl50-cpp <clang-tidy/checks/cert/dcl50-cpp>` to
  :doc:`modernize-avoid-variadic-functions
  <clang-tidy/checks/modernize/avoid-variadic-functions>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-dcl58-cpp <clang-tidy/checks/cert/dcl58-cpp>` to
  :doc:`bugprone-std-namespace-modification
  <clang-tidy/checks/bugprone/std-namespace-modification>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-env33-c <clang-tidy/checks/cert/env33-c>` to
  :doc:`bugprone-command-processor
  <clang-tidy/checks/bugprone/command-processor>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-err34-c <clang-tidy/checks/cert/err34-c>` to
  :doc:`bugprone-unchecked-string-to-number-conversion
  <clang-tidy/checks/bugprone/unchecked-string-to-number-conversion>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-err52-cpp <clang-tidy/checks/cert/err52-cpp>` to
  :doc:`modernize-avoid-setjmp-longjmp
  <clang-tidy/checks/modernize/avoid-setjmp-longjmp>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-err58-cpp <clang-tidy/checks/cert/err58-cpp>` to
  :doc:`bugprone-throwing-static-initialization
  <clang-tidy/checks/bugprone/throwing-static-initialization>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-err60-cpp <clang-tidy/checks/cert/err60-cpp>` to
  :doc:`bugprone-exception-copy-constructor-throws
  <clang-tidy/checks/bugprone/exception-copy-constructor-throws>`

- Renamed :doc:`cert-flp30-c <clang-tidy/checks/cert/flp30-c>` to
  :doc:`bugprone-float-loop-counter
  <clang-tidy/checks/bugprone/float-loop-counter>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-mem57-cpp <clang-tidy/checks/cert/mem57-cpp>` to
  :doc:`bugprone-default-operator-new-on-overaligned-type
  <clang-tidy/checks/bugprone/default-operator-new-on-overaligned-type>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-msc30-c <clang-tidy/checks/cert/msc30-c>` to
  :doc:`misc-predictable-rand
  <clang-tidy/checks/misc/predictable-rand>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-msc32-c <clang-tidy/checks/cert/msc32-c>` to
  :doc:`bugprone-random-generator-seed
  <clang-tidy/checks/bugprone/random-generator-seed>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-msc50-cpp <clang-tidy/checks/cert/msc50-cpp>` to
  :doc:`misc-predictable-rand
  <clang-tidy/checks/misc/predictable-rand>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-msc51-cpp <clang-tidy/checks/cert/msc51-cpp>` to
  :doc:`bugprone-random-generator-seed
  <clang-tidy/checks/bugprone/random-generator-seed>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-oop57-cpp <clang-tidy/checks/cert/oop57-cpp>` to
  :doc:`bugprone-raw-memory-call-on-non-trivial-type
  <clang-tidy/checks/bugprone/raw-memory-call-on-non-trivial-type>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`cert-oop58-cpp <clang-tidy/checks/cert/oop58-cpp>` to
  :doc:`bugprone-copy-constructor-mutates-argument
  <clang-tidy/checks/bugprone/copy-constructor-mutates-argument>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`fuchsia-multiple-inheritance <clang-tidy/checks/fuchsia/multiple-inheritance>` to
  :doc:`misc-multiple-inheritance
  <clang-tidy/checks/misc/multiple-inheritance>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`google-build-namespaces <clang-tidy/checks/google/build-namespaces>` to
  :doc:`misc-anonymous-namespace-in-header
  <clang-tidy/checks/misc/anonymous-namespace-in-header>`
  keeping initial check as an alias to the new one.

- Renamed :doc:`google-readability-casting <clang-tidy/checks/google/readability-casting>` to
  :doc:`modernize-avoid-c-style-cast
  <clang-tidy/checks/modernize/avoid-c-style-cast>`
  keeping initial check as an alias to the new one.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-chained-comparison
  <clang-tidy/checks/bugprone/chained-comparison>` check by adding a
  new option `IgnoreMacros` to suppress warnings within macro
  expansions.

- Improved :doc:`bugprone-dynamic-static-initializers
  <clang-tidy/checks/bugprone/dynamic-static-initializers>` check by
  avoiding false positives for ``constexpr`` variables whose
  initializers are value-dependent.

- Improved :doc:`bugprone-easily-swappable-parameters
  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
  correcting a spelling mistake on its option
  ``NamePrefixSuffixSilenceDissimilarityTreshold``.

- Improved :doc:`bugprone-exception-escape
  <clang-tidy/checks/bugprone/exception-escape>` check:

  - Exceptions from lambda captures are now diagnosed, exceptions in the
    bodies of lambdas that aren't actually invoked are not.

  - Fixed an issue where the check wouldn't diagnose throws in arguments to
    functions or constructors.

  - Added fine-grained configuration via options `CheckDestructors`,
    `CheckMoveMemberFunctions`, `CheckMain`, `CheckedSwapFunctions`, and
    `CheckNothrowFunctions`.

- Improved :doc:`bugprone-infinite-loop
  <clang-tidy/checks/bugprone/infinite-loop>` check by adding detection for
  variables introduced by structured bindings.

- Improved :doc:`bugprone-invalid-enum-default-initialization
  <clang-tidy/checks/bugprone/invalid-enum-default-initialization>` with new
  `IgnoredEnums` option to ignore specified enums during analysis.

- Improved :doc:`bugprone-macro-parentheses
  <clang-tidy/checks/bugprone/macro-parentheses>` check by fixing false
  positives when using C++ template parameters.

- Improved :doc:`bugprone-narrowing-conversions
  <clang-tidy/checks/bugprone/narrowing-conversions>` check by fixing
  false positive from analysis of a conditional expression in C.

- Improved :doc:`bugprone-not-null-terminated-result
  <clang-tidy/checks/bugprone/not-null-terminated-result>` check by fixing
  bogus fix-its for ``strncmp`` and ``wcsncmp`` on Windows and
  a crash caused by certain value-dependent expressions.

- Improved :doc:`bugprone-reserved-identifier
  <clang-tidy/checks/bugprone/reserved-identifier>` check by ignoring
  declarations and macros in system headers.

- Improved :doc:`bugprone-signed-char-misuse
  <clang-tidy/checks/bugprone/signed-char-misuse>` check by fixing
  false positives on C23 enums with the fixed underlying type of signed char.

- Improved :doc:`bugprone-sizeof-expression
  <clang-tidy/checks/bugprone/sizeof-expression>` check by fixing
  a crash on ``sizeof`` of an array of dependent type.

- Improved :doc:`bugprone-suspicious-include
  <clang-tidy/checks/bugprone/suspicious-include>` check by adding
  `IgnoredRegex` option.

- Improved :doc:`bugprone-tagged-union-member-count
  <clang-tidy/checks/bugprone/tagged-union-member-count>` by fixing a false
  positive when enums or unions from system header files or the ``std``
  namespace are treated as the tag or the data part of a user-defined
  tagged union respectively.

- Improved :doc:`bugprone-throw-keyword-missing
  <clang-tidy/checks/bugprone/throw-keyword-missing>` check:

  - Now it only considers the canonical types of base classes as written.

  - Added a note on the base class that triggered the warning.

  - Fixed an issue where the check wouldn't fire in constructors or
    (in certain contexts) lambdas.

- Improved :doc:`bugprone-unchecked-optional-access
  <clang-tidy/checks/bugprone/unchecked-optional-access>` check:

  - Added support for ``NullableValue::makeValue`` and
    ``NullableValue::makeValueInplace`` to prevent false-positives for
    ``BloombergLP::bdlb::NullableValue``.

  - Fixed false-positives for ``bsl::optional`` containing allocator-aware type.

  - Added the `IgnoreValueCalls` option to suppress diagnostics for
    ``optional::value()``.

  - Added the `IgnoreSmartPointerDereference` option to ignore optionals
    reached via smart-pointer-like dereference, while still diagnosing UB-prone
    dereferences via ``operator*`` and ``operator->``.

- Improved :doc:`bugprone-unhandled-self-assignment
  <clang-tidy/checks/bugprone/unhandled-self-assignment>` check by adding
  an additional matcher that generalizes the copy-and-swap idiom pattern
  detection.

- Improved :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` check by hiding the default
  suffix when the reason starts with the character `>` in the `CustomFunctions`
  option.

- Improved :doc:`bugprone-use-after-move
  <clang-tidy/checks/bugprone/use-after-move>` check:

  - Added `InvalidationFunctions` option to support custom invalidation functions.

  - Added `ReinitializationFunctions` option to support custom reinitialization
    functions.

- Improved :doc:`cppcoreguidelines-avoid-non-const-global-variables
  <clang-tidy/checks/cppcoreguidelines/avoid-non-const-global-variables>` check
  by adding a new option `AllowThreadLocal` that suppresses warnings on
  non-const global variables with thread-local storage duration.

- Improved :doc:`cppcoreguidelines-init-variables
  <clang-tidy/checks/cppcoreguidelines/init-variables>` check by fixing the
  insertion location for function pointers with multiple parameters.

- Improved :doc:`cppcoreguidelines-macro-usage
  <clang-tidy/checks/cppcoreguidelines/macro-usage>` check by excluding macro
  bodies that starts with ``__attribute__((..))`` keyword.
  Such a macro body is unlikely a proper expression and so suggesting users
  an impossible rewrite into a template function should be avoided.

- Improved :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines/prefer-member-initializer>` check to
  avoid false positives on inherited members in class templates.

- Improved :doc:`cppcoreguidelines-pro-bounds-pointer-arithmetic
  <clang-tidy/checks/cppcoreguidelines/pro-bounds-pointer-arithmetic>` check
  adding `AllowIncrementDecrementOperators` option to allow pointer arithmetic
  via prefix/postfix increment or decrement operators.

- Improved :doc:`cppcoreguidelines-pro-type-member-init
  <clang-tidy/checks/cppcoreguidelines/pro-type-member-init>` check to
  correctly ignore ``std::array`` and other array-like containers when
  `IgnoreArrays` option is set to `true`.

- Improved :doc:`fuchsia-multiple-inheritance
  <clang-tidy/checks/fuchsia/multiple-inheritance>`
  by fixing an issue where the check would only analyze the first class with
  a given name in the program, missing any subsequent classes with that same
  name (declared in a different scope).

- Improved :doc:`google-readability-casting
  <clang-tidy/checks/google/readability-casting>` check by adding fix-it
  notes for downcasts and casts to void pointer.

- Improved :doc:`google-readability-todo
  <clang-tidy/checks/google/readability-todo>` check to accept the new TODO
  format from the Google Style Guide.

- Improved :doc:`llvm-prefer-isa-or-dyn-cast-in-conditionals
  <clang-tidy/checks/llvm/prefer-isa-or-dyn-cast-in-conditionals>` check:

  - Fix-it handles callees with nested-name-specifier correctly.

  - ``if`` statements with init-statement (``if (auto X = ...; ...)``) are
    handled correctly.

  - ``for`` loops are supported.

- Improved :doc:`misc-const-correctness
  <clang-tidy/checks/misc/const-correctness>` check:

  - Fixed false positives when pointers is transferred to non-const references.

  - Fixed false positives of function pointer, on return of non-const pointer,
    on pointer-to-member operator.

  - Fixed false positives when the address of a variable is taken to be passed
    to a function.

- Improved :doc:`misc-coroutine-hostile-raii
  <clang-tidy/checks/misc/coroutine-hostile-raii>` check by adding the option
  `AllowedCallees`, that allows exempting safely awaitable callees from the
  check.

- Improved :doc:`misc-header-include-cycle
  <clang-tidy/checks/misc/header-include-cycle>` check performance.

- Improved :doc:`misc-use-internal-linkage
  <clang-tidy/checks/misc/use-internal-linkage>` check:

  - Added suggestion of internal linkage for user-defined types (structs,
    classes, unions, and enums).

  - Added fine-grained options `AnalyzeFunctions`, `AnalyzeVariables`,
    `AnalyzeTypes` to control whether the check should diagnose functions,
    variables, and/or user-defined types.

  - Enabled the check for C.

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
  <clang-tidy/checks/readability/identifier-naming>` check:

  - Now ignores declarations and macros in system headers.

  - The documentation is improved to differentiate the general options from the
    specific ones.

  - Options for fine-grained control over ``constexpr`` variables were added.

  - Added default options which simplify configs by removing the need to specify
    each identifier kind separately.

- Improved :doc:`readability-implicit-bool-conversion
  <clang-tidy/checks/readability/implicit-bool-conversion>` check:

  - Now correctly adds parentheses when inner expressions are implicitly
    converted multiple times.

  - Enabled the check for C99 and later standards.

  - Now allows implicit conversions from ``bool`` to integer when used as
    operands of logical operators (``&&``, ``||``) in C.

- Improved :doc:`readability-inconsistent-declaration-parameter-name
  <clang-tidy/checks/readability/inconsistent-declaration-parameter-name>` check
  by not enforcing parameter name consistency between a variadic parameter pack
  in the primary template and specific parameters in its specializations.

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
