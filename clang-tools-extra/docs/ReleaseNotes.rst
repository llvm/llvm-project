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

- New global configuration file options `HeaderFileExtensions` and
  `ImplementationFileExtensions`, replacing the check-local options of the
  same name.

- Support specifying `Checks` as a YAML list in the `.clang-tidy` configuration
  file.

- Fix a potential crash when using the `--dump-config` option.

New checks
^^^^^^^^^^

- New :doc:`bugprone-multiple-new-in-one-expression
  <clang-tidy/checks/bugprone/multiple-new-in-one-expression>` check.

  Finds multiple ``new`` operator calls in a single expression, where the allocated
  memory by the first ``new`` may leak if the second allocation fails and throws exception.

- New :doc:`bugprone-non-zero-enum-to-bool-conversion
  <clang-tidy/checks/bugprone/non-zero-enum-to-bool-conversion>` check.

  Detect implicit and explicit casts of ``enum`` type into ``bool`` where ``enum`` type
  doesn't have a zero-value enumerator.

- New :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` check.

  Checks for functions that have safer, more secure replacements available, or
  are considered deprecated due to design flaws.
  This check relies heavily on, but is not exclusive to, the functions from
  the *Annex K. "Bounds-checking interfaces"* of C11.

- New :doc:`cppcoreguidelines-avoid-capturing-lambda-coroutines
  <clang-tidy/checks/cppcoreguidelines/avoid-capturing-lambda-coroutines>` check.

  Flags C++20 coroutine lambdas with non-empty capture lists that may cause
  use-after-free errors and suggests avoiding captures or ensuring the lambda
  closure object has a guaranteed lifetime.

- New :doc:`cppcoreguidelines-misleading-capture-default-by-value
  <clang-tidy/checks/cppcoreguidelines/misleading-capture-default-by-value>` check.

  Warns when lambda specify a by-value capture default and capture ``this``.

- New :doc:`cppcoreguidelines-rvalue-reference-param-not-moved
  <clang-tidy/checks/cppcoreguidelines/rvalue-reference-param-not-moved>` check.

  Warns when an rvalue reference function parameter is never moved within
  the function body.

- New :doc:`llvmlibc-inline-function-decl
  <clang-tidy/checks/llvmlibc/inline-function-decl>` check.

  Checks that all implicit and explicit inline functions in header files are
  tagged with the ``LIBC_INLINE`` macro.

- New :doc:`modernize-type-traits
  <clang-tidy/checks/modernize/type-traits>` check.

  Converts standard library type traits of the form ``traits<...>::type`` and
  ``traits<...>::value`` into ``traits_t<...>`` and ``traits_v<...>`` respectively.

- New :doc:`performance-avoid-endl
  <clang-tidy/checks/performance/avoid-endl>` check.

  Finds uses of ``std::endl`` on streams and replaces them with ``'\n'``.

- New :doc:`readability-avoid-unconditional-preprocessor-if
  <clang-tidy/checks/readability/avoid-unconditional-preprocessor-if>` check.

  Finds code blocks that are constantly enabled or disabled in preprocessor
  directives by analyzing ``#if`` conditions, such as ``#if 0`` and
  ``#if 1``, etc.

- New :doc:`readability-operators-representation
  <clang-tidy/checks/readability/operators-representation>` check.

  Enforces consistent token representation for invoked binary, unary and
  overloaded operators in C++ code.

New check aliases
^^^^^^^^^^^^^^^^^

- New alias :doc:`cert-msc24-c
  <clang-tidy/checks/cert/msc24-c>` to :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` was added.

- New alias :doc:`cert-msc33-c
  <clang-tidy/checks/cert/msc33-c>` to :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` was added.

- New alias :doc:`cppcoreguidelines-use-default-member-init
  <clang-tidy/checks/cppcoreguidelines/use-default-member-init>` to
  :doc:`modernize-use-default-member-init
  <clang-tidy/checks/modernize/use-default-member-init>` was added.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Improved :doc:`readability-redundant-string-cstr
  <clang-tidy/checks/readability/redundant-string-cstr>` check to recognise
  unnecessary ``std::string::c_str()`` and ``std::string::data()`` calls in
  arguments to ``std::print``, ``std::format`` or other functions listed in
  the ``StringParameterFunction`` check option.

- Improved :doc:`bugprone-dangling-handle
  <clang-tidy/checks/bugprone/dangling-handle>` check enhancing detection of
  handles behind type aliases.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`bugprone-dynamic-static-initializers
  <clang-tidy/checks/bugprone/dynamic-static-initializers>` check.
  Global options of the same name should be used instead.

- Improved :doc:`bugprone-exception-escape
  <clang-tidy/checks/bugprone/exception-escape>` to not emit warnings for
  forward declarations of functions.

- Improved :doc:`bugprone-fold-init-type
  <clang-tidy/checks/bugprone/fold-init-type>` to handle iterators that do not
  define `value_type` type aliases.

- Improved :doc:`bugprone-incorrect-roundings
  <clang-tidy/checks/bugprone/incorrect-roundings>` check by adding support for
  other floating point representations in float constant like ``0.5L``.

- Deprecated check-local options `HeaderFileExtensions` and `ImplementationFileExtensions`
  in :doc:`bugprone-suspicious-include
  <clang-tidy/checks/bugprone/suspicious-include>` check.
  Global options of the same name should be used instead.

- Improved :doc:`bugprone-unchecked-optional-access
  <clang-tidy/checks/bugprone/unchecked-optional-access>` check to properly handle calls
  to ``std::forward``.

- Improved :doc:`bugprone-use-after-move
  <clang-tidy/checks/bugprone/use-after-move>` check. Detect uses and moves in
  constructor initializers. Correctly handle constructor arguments as being
  sequenced when constructor call is written as list-initialization.

- Deprecated :doc:`cert-dcl21-cpp
  <clang-tidy/checks/cert/dcl21-cpp>` check.

- Deprecated C.48 enforcement from :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines/prefer-member-initializer>`. Please use
  :doc:`cppcoreguidelines-use-default-member-init
  <clang-tidy/checks/cppcoreguidelines/use-default-member-init>` instead.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`google-build-namespaces
  <clang-tidy/checks/google/build-namespaces>` check.
  Global options of the same name should be used instead.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`google-global-names-in-headers
  <clang-tidy/checks/google/global-names-in-headers>` check.
  Global options of the same name should be used instead.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`llvm-header-guard
  <clang-tidy/checks/llvm/header-guard>` check.
  Global options of the same name should be used instead.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`misc-definitions-in-headers
  <clang-tidy/checks/misc/definitions-in-headers>` check.
  Global options of the same name should be used instead.

- Fixed false positive in :doc:`misc-definitions-in-headers
  <clang-tidy/checks/misc/definitions-in-headers>` to avoid warning on
  declarations inside anonymous namespaces.

- Improved :doc:`misc-unused-parameters
  <clang-tidy/checks/misc/unused-parameters>` check with new `IgnoreVirtual`
  option to optionally ignore virtual methods.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`misc-unused-using-decls
  <clang-tidy/checks/misc/unused-using-decls>` check.
  Global options of the same name should be used instead.

- In :doc:`modernize-use-default-member-init
  <clang-tidy/checks/modernize/use-default-member-init>` count template
  constructors toward hand written constructors so that they are skipped if more
  than one exists.

- Fixed false positive in :doc:`modernize-use-equals-default
  <clang-tidy/checks/modernize/use-equals-default>` check for special member
  functions containing macros or preprocessor directives, and out-of-line special
  member functions in unions.

- Improved :doc:`modernize-use-override
  <clang-tidy/checks/modernize/use-override>` check with new
  `IgnoreTemplateInstantiations` option to optionally ignore virtual function
  overrides that are part of template instantiations.

- Fixed reading `HungarianNotation.CString.*` options in
  :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check.

- Renamed `HungarianNotation.CString` options `CharPrinter` and
  `WideCharPrinter` to `CharPointer` and `WideCharPointer` respectively in
  :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check.

- Updated the Hungarian prefixes for enums in C files to match those used in C++
  files for improved readability, as checked by :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>`. To preserve the previous
  behavior of using `i` as the prefix for enum tags, set the `EnumConstantPrefix`
  option to `i` instead of using `EnumConstantHungarianPrefix`.

- Fixed a hungarian notation issue in :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` which failed to indicate
  the number of asterisks.

- Fixed a false positive in :doc:`readability-implicit-bool-conversion
  <clang-tidy/checks/readability/implicit-bool-conversion>` check warning would
  be unnecessarily emitted for explicit cast using direct list initialization.

- Added support to optionally ignore user-defined literals in
  :doc:`readability-magic-numbers<clang-tidy/checks/readability/magic-numbers>`.

- Fixed a false positive in :doc:`readability-container-size-empty
  <clang-tidy/checks/readability/container-size-empty>` check when comparing
  ``std::array`` objects to default constructed ones. The behavior for this and
  other relevant classes can now be configured with a new option.

- Improved :doc:`bugprone-too-small-loop-variable
  <clang-tidy/checks/bugprone/too-small-loop-variable>` check. Basic support
  for bit-field and integer members as a loop variable or upper limit were added.

- Improved :doc:`readability-magic-numbers
  <clang-tidy/checks/readability/magic-numbers>` check, now allows for
  magic numbers in type aliases such as ``using`` and ``typedef`` declarations if
  the new ``IgnoreTypeAliases`` option is set to true.

- Fixed a false positive in :doc:`readability-misleading-indentation
  <clang-tidy/checks/readability/misleading-indentation>` check when warning would
  be unnecessarily emitted for template dependent ``if constexpr``.

- Fixed incorrect fixes in :doc:`readability-redundant-declaration
  <clang-tidy/checks/readability/redundant-declaration>` check when linkage
  (like ``extern "C"``) is explicitly specified.

- Improved :doc:`readability-static-accessed-through-instance
  <clang-tidy/checks/readability/static-accessed-through-instance>` check to
  support unscoped enumerations through instances and fixed usage of anonymous
  structs or classes.

- Fixed a false positive in :doc:`cppcoreguidelines-slicing
  <clang-tidy/checks/cppcoreguidelines/slicing>` check when warning would be
  emitted in constructor for virtual base class initialization.

- Improved :doc:`bugprone-use-after-move
  <clang-tidy/checks/bugprone/use-after-move>` to understand that there is a
  sequence point between designated initializers.

- Fixed an issue in the :doc:`performance-noexcept-move-constructor
  <clang-tidy/checks/performance/noexcept-move-constructor>` checker that was causing
  false-positives when the move constructor or move assign operator were defaulted.

- Fixed an issue in :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` when specifying an empty
  string for ``Prefix`` or ``Suffix`` options could result in the style not
  being used.

- Fixed an issue in :doc:`google-readability-avoid-underscore-in-googletest-name
  <clang-tidy/checks/google/readability-avoid-underscore-in-googletest-name>` when using
  ``DISABLED_`` in the test suite name.

- Improved :doc:`modernize-concat-nested-namespaces
  <clang-tidy/checks/modernize/concat-nested-namespaces>` to fix incorrect fixes when
  using macro between namespace declarations, to fix false positive when using namespace
  with attributes and to support nested inline namespace introduced in c++20.

- Fixed a false positive in :doc:`performance-no-automatic-move
  <clang-tidy/checks/performance/no-automatic-move>` when warning would be
  emitted for a const local variable to which NRVO is applied.

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
