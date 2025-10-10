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

- Renamed a few :program:`clang-tidy` check options, as they
  were misspelled:

  - `NamePrefixSuffixSilenceDissimilarityTreshold` to
    `NamePrefixSuffixSilenceDissimilarityThreshold` in
    :doc:`bugprone-easily-swappable-parameters
    <clang-tidy/checks/bugprone/easily-swappable-parameters>`

  - `CharTypdefsToIgnore` to `CharTypedefsToIgnore` in
    :doc:`bugprone-signed-char-misuse
    <clang-tidy/checks/bugprone/signed-char-misuse>`

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

- :program:`clang-tidy` no longer attemps to analyze code from system headers
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

- Deprecated the :program:`clang-tidy` ``zircon`` module. All checks have been
  moved to the ``fuchsia`` module instead. The ``zircon`` module will be removed
  in the 24th release.

New checks
^^^^^^^^^^

- New :doc:`bugprone-invalid-enum-default-initialization
  <clang-tidy/checks/bugprone/invalid-enum-default-initialization>` check.

  Detects default initialization (to 0) of variables with ``enum`` type where
  the enum has no enumerator with value of 0.

- New :doc:`bugprone-derived-method-shadowing-base-method
  <clang-tidy/checks/bugprone/derived-method-shadowing-base-method>` check.

  Finds derived class methods that shadow a (non-virtual) base class method.

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

- New :doc:`readability-redundant-parentheses
  <clang-tidy/checks/readability/redundant-parentheses>` check.

  Detect redundant parentheses.

New check aliases
^^^^^^^^^^^^^^^^^

- Renamed :doc:`cert-dcl50-cpp <clang-tidy/checks/cert/dcl50-cpp>` to
  :doc:`modernize-avoid-variadic-functions
  <clang-tidy/checks/modernize/avoid-variadic-functions>`
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

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-easily-swappable-parameters
  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
  correcting a spelling mistake on its option
  ``NamePrefixSuffixSilenceDissimilarityTreshold``.

- Improved :doc:`bugprone-exception-escape
  <clang-tidy/checks/bugprone/exception-escape>` check's handling of lambdas:
  exceptions from captures are now diagnosed, exceptions in the bodies of
  lambdas that aren't actually invoked are not.

- Improved :doc:`bugprone-infinite-loop
  <clang-tidy/checks/bugprone/infinite-loop>` check by adding detection for
  variables introduced by structured bindings.

- Improved :doc:`bugprone-invalid-enum-default-initialization
  <clang-tidy/checks/bugprone/invalid-enum-default-initialization>` with new
  `IgnoredEnums` option to ignore specified enums during analysis.

- Improved :doc:`bugprone-narrowing-conversions
  <clang-tidy/checks/bugprone/narrowing-conversions>` check by fixing
  false positive from analysis of a conditional expression in C.

- Improved :doc:`bugprone-reserved-identifier
  <clang-tidy/checks/bugprone/reserved-identifier>` check by ignoring
  declarations and macros in system headers.

- Improved :doc:`bugprone-signed-char-misuse
  <clang-tidy/checks/bugprone/signed-char-misuse>` check by fixing
  false positives on C23 enums with the fixed underlying type of signed char.

- Improved :doc:`bugprone-sizeof-expression
  <clang-tidy/checks/bugprone/sizeof-expression>` check by fixing
  a crash on ``sizeof`` of an array of dependent type.

- Improved :doc:`bugprone-tagged-union-member-count
  <clang-tidy/checks/bugprone/tagged-union-member-count>` by fixing a false
  positive when enums or unions from system header files or the ``std``
  namespace are treated as the tag or the data part of a user-defined
  tagged union respectively.

- Improved :doc:`bugprone-throw-keyword-missing
  <clang-tidy/checks/bugprone/throw-keyword-missing>` check by only considering
  the canonical types of base classes as written and adding a note on the base
  class that triggered the warning.

- Improved :doc:`bugprone-unchecked-optional-access
  <clang-tidy/checks/bugprone/unchecked-optional-access>` check by supporting
  ``NullableValue::makeValue`` and ``NullableValue::makeValueInplace`` to
  prevent false-positives for ``BloombergLP::bdlb::NullableValue`` type.

- Improved :doc:`bugprone-unhandled-self-assignment
  <clang-tidy/checks/bugprone/unhandled-self-assignment>` check by adding
  an additional matcher that generalizes the copy-and-swap idiom pattern
  detection.

- Improved :doc:`cppcoreguidelines-init-variables
  <clang-tidy/checks/cppcoreguidelines/init-variables>` check by fixing the
  insertion location for function pointers with multiple parameters.

- Improved :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines/prefer-member-initializer>` check to
  avoid false positives on inherited members in class templates.

- Improved :doc:`cppcoreguidelines-pro-bounds-pointer-arithmetic
  <clang-tidy/checks/cppcoreguidelines/pro-bounds-pointer-arithmetic>` check
  adding an option to allow pointer arithmetic via prefix/postfix increment or
  decrement operators.

- Improved :doc:`llvm-prefer-isa-or-dyn-cast-in-conditionals
  <clang-tidy/checks/llvm/prefer-isa-or-dyn-cast-in-conditionals>` check:

  - Fix-it handles callees with nested-name-specifier correctly.

  - ``if`` statements with init-statement (``if (auto X = ...; ...)``) are
    handled correctly.

  - ``for`` loops are supported.

- Improved :doc:`misc-const-correctness
  <clang-tidy/checks/misc/const-correctness>` check to avoid false
  positives when pointers is tranferred to non-const references 
  and avoid false positives of function pointer.

- Improved :doc:`misc-header-include-cycle
  <clang-tidy/checks/misc/header-include-cycle>` check performance.

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

- Improved :doc:`modernize-use-nullptr
  <clang-tidy/checks/modernize/use-nullptr>` check by fixing a crash
  on Windows when the check was enabled with a 32-bit :program:`clang-tidy`
  binary.

- Improved :doc:`modernize-use-std-format
  <clang-tidy/checks/modernize/use-std-format>` check to correctly match
  when the format string is converted to a different type by an implicit
  constructor call.

- Improved :doc:`modernize-use-std-print
  <clang-tidy/checks/modernize/use-std-print>` check to correctly match
  when the format string is converted to a different type by an implicit
  constructor call.

- Improved :doc:`performance-unnecessary-copy-initialization
  <clang-tidy/checks/performance/unnecessary-copy-initialization>` by printing
  the type of the diagnosed variable.

- Improved :doc:`performance-unnecessary-value-param
  <clang-tidy/checks/performance/unnecessary-value-param>` by printing
  the type of the diagnosed variable.

- Improved :doc:`portability-template-virtual-member-function
  <clang-tidy/checks/portability/template-virtual-member-function>` check to
  avoid false positives on pure virtual member functions.

- Improved :doc:`readability-container-contains
  <clang-tidy/checks/readability/container-contains>` to support string
  comparisons to ``npos``. Internal changes may cause new rare false positives
  in non-standard containers.

- Improved :doc:`readability-container-size-empty
  <clang-tidy/checks/readability/container-size-empty>` check by correctly
  generating fix-it hints when size method is called from implicit ``this``,
  ignoring default constructors with user provided arguments and adding
  detection in container's method except ``empty``.

- Improved :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check by ignoring
  declarations and macros in system headers. The documentation is also improved
  to differentiate the general options from the specific ones.

- Improved :doc:`readability-qualified-auto
  <clang-tidy/checks/readability/qualified-auto>` check by adding the option
  `IgnoreAliasing`, that allows not looking at underlying types of type aliases.

- Improved :doc:`readability-uppercase-literal-suffix
  <clang-tidy/checks/readability/uppercase-literal-suffix>` check to recognize
  literal suffixes added in C++23 and C23.

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
