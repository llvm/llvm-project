===================================================
Extra Clang Tools 8.0.0 (In-Progress) Release Notes
===================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 8 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 8.0.0. Here we describe the status of the Extra Clang Tools in
some detail, including major improvements from the previous release and new
feature work. All LLVM releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or
the `LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Subversion checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Extra Clang Tools 8.0.0?
======================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

...

Improvements to clangd
----------------------

The improvements are...

Improvements to clang-doc
-------------------------

The improvements are...

Improvements to clang-query
---------------------------

- A new command line parameter ``--preload`` was added to
  run commands from a file and then start the interactive interpreter.

- The command ``q`` can was added as an alias for ``quit`` to exit the
  ``clang-query`` interpreter.

- It is now possible to bind to named values (the result of ``let``
  expressions). For example:

  .. code-block:: none

    let fn functionDecl()
    match fn.bind("foo")

- It is now possible to write comments in ``clang-query`` code. This
  is primarily useful when using script-mode. Comments are all content
  following the ``#`` character on a line:

  .. code-block:: none

    # This is a comment
    match fn.bind("foo") # This is a trailing comment

- The new ``set print-matcher true`` command now causes ``clang-query`` to
  print the evaluated matcher together with the resulting bindings.

- A new output mode ``detailed-ast`` was added to ``clang-query``. The
  existing ``dump`` output mode is now a deprecated alias
  for ``detailed-ast``

- Output modes can now be enabled or disabled non-exclusively.  For example,

  .. code-block:: none

    # Enable detailed-ast without disabling other output, such as diag
    enable output detailed-ast
    m functionDecl()

    # Disable detailed-ast only
    disable output detailed-ast
    m functionDecl()

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- New :doc:`abseil-duration-comparison
  <clang-tidy/checks/abseil-duration-comparison>` check.

  Checks for comparisons which should be done in the ``absl::Duration`` domain
  instead of the float of integer domains.

- New :doc:`abseil-duration-division
  <clang-tidy/checks/abseil-duration-division>` check.

  Checks for uses of ``absl::Duration`` division that is done in a
  floating-point context, and recommends the use of a function that
  returns a floating-point value.

- New :doc:`abseil-duration-factory-float
  <clang-tidy/checks/abseil-duration-factory-float>` check.

  Checks for cases where the floating-point overloads of various
  ``absl::Duration`` factory functions are called when the more-efficient
  integer versions could be used instead.

- New :doc:`abseil-duration-factory-scale
  <clang-tidy/checks/abseil-duration-factory-scale>` check.

  Checks for cases where arguments to ``absl::Duration`` factory functions are
  scaled internally and could be changed to a different factory function.

- New :doc:`abseil-duration-subtraction
  <clang-tidy/checks/abseil-duration-subtraction>` check.

  Checks for cases where subtraction should be performed in the
  ``absl::Duration`` domain.

- New :doc:`abseil-faster-strsplit-delimiter
  <clang-tidy/checks/abseil-faster-strsplit-delimiter>` check.

  Finds instances of ``absl::StrSplit()`` or ``absl::MaxSplits()`` where the
  delimiter is a single character string literal and replaces with a character.

- New :doc:`abseil-no-internal-dependencies
  <clang-tidy/checks/abseil-no-internal-dependencies>` check.

  Gives a warning if code using Abseil depends on internal details.

- New :doc:`abseil-no-namespace
  <clang-tidy/checks/abseil-no-namespace>` check.

  Ensures code does not open ``namespace absl`` as that violates Abseil's
  compatibility guidelines.

- New :doc:`abseil-redundant-strcat-calls
  <clang-tidy/checks/abseil-redundant-strcat-calls>` check.

  Suggests removal of unnecessary calls to ``absl::StrCat`` when the result is
  being passed to another ``absl::StrCat`` or ``absl::StrAppend``.

- New :doc:`abseil-str-cat-append
  <clang-tidy/checks/abseil-str-cat-append>` check.

  Flags uses of ``absl::StrCat()`` to append to a ``std::string``. Suggests
  ``absl::StrAppend()`` should be used instead.

- New :doc:`abseil-upgrade-duration-conversions
  <clang-tidy/checks/abseil-upgrade-duration-conversions>` check.

  Finds calls to ``absl::Duration`` arithmetic operators and factories whose
  argument needs an explicit cast to continue compiling after upcoming API
  changes.

- New :doc:`bugprone-too-small-loop-variable
  <clang-tidy/checks/bugprone-too-small-loop-variable>` check.

  Detects those ``for`` loops that have a loop variable with a "too small" type
  which means this type can't represent all values which are part of the
  iteration range.

- New :doc:`cppcoreguidelines-macro-usage
  <clang-tidy/checks/cppcoreguidelines-macro-usage>` check.

  Finds macro usage that is considered problematic because better language
  constructs exist for the task.

- New :doc:`google-objc-function-naming
  <clang-tidy/checks/google-objc-function-naming>` check.

  Checks that function names in function declarations comply with the naming
  conventions described in the Google Objective-C Style Guide.

- New :doc:`misc-non-private-member-variables-in-classes
  <clang-tidy/checks/misc-non-private-member-variables-in-classes>` check.

  Finds classes that not only contain the data (non-static member variables),
  but also have logic (non-static member functions), and diagnoses all member
  variables that have any other scope other than ``private``.

- New :doc:`modernize-avoid-c-arrays
  <clang-tidy/checks/modernize-avoid-c-arrays>` check.

  Finds C-style array types and recommend to use ``std::array<>`` /
  ``std::vector<>``.

- New :doc:`modernize-concat-nested-namespaces
  <clang-tidy/checks/modernize-concat-nested-namespaces>` check.

  Checks for uses of nested namespaces in the form of
  ``namespace a { namespace b { ... }}`` and offers change to
  syntax introduced in C++17 standard: ``namespace a::b { ... }``.

- New :doc:`modernize-deprecated-ios-base-aliases
  <clang-tidy/checks/modernize-deprecated-ios-base-aliases>` check.

  Detects usage of the deprecated member types of ``std::ios_base`` and replaces
  those that have a non-deprecated equivalent.

- New :doc:`modernize-use-nodiscard
  <clang-tidy/checks/modernize-use-nodiscard>` check.

  Adds ``[[nodiscard]]`` attributes (introduced in C++17) to member functions
  to highlight at compile time which return values should not be ignored.

- New :doc:`readability-isolate-decl
  <clang-tidy/checks/readability-isolate-declaration>` check.

  Detects local variable declarations declaring more than one variable and
  tries to refactor the code to one statement per declaration.

- New :doc:`readability-const-return-type
  <clang-tidy/checks/readability-const-return-type>` check.

  Checks for functions with a ``const``-qualified return type and recommends
  removal of the ``const`` keyword.

- New :doc:`readability-magic-numbers
  <clang-tidy/checks/readability-magic-numbers>` check.

  Detects usage of magic numbers, numbers that are used as literals instead of
  introduced via constants or symbols.

- New :doc:`readability-uppercase-literal-suffix
  <clang-tidy/checks/readability-uppercase-literal-suffix>` check.

  Detects when the integral literal or floating point literal has non-uppercase
  suffix, and suggests to make the suffix uppercase. The list of destination
  suffixes can be optionally provided.

- New alias :doc:`cert-dcl16-c
  <clang-tidy/checks/cert-dcl16-c>` to :doc:`readability-uppercase-literal-suffix
  <clang-tidy/checks/readability-uppercase-literal-suffix>`
  added.

- New alias :doc:`cppcoreguidelines-avoid-c-arrays
  <clang-tidy/checks/cppcoreguidelines-avoid-c-arrays>`
  to :doc:`modernize-avoid-c-arrays
  <clang-tidy/checks/modernize-avoid-c-arrays>` added.

- New alias :doc:`cppcoreguidelines-non-private-member-variables-in-classes
  <clang-tidy/checks/cppcoreguidelines-non-private-member-variables-in-classes>`
  to :doc:`misc-non-private-member-variables-in-classes
  <clang-tidy/checks/misc-non-private-member-variables-in-classes>`
  added.

- New alias :doc:`hicpp-avoid-c-arrays
  <clang-tidy/checks/hicpp-avoid-c-arrays>`
  to :doc:`modernize-avoid-c-arrays
  <clang-tidy/checks/modernize-avoid-c-arrays>` added.

- New alias :doc:`hicpp-uppercase-literal-suffix
  <clang-tidy/checks/hicpp-uppercase-literal-suffix>` to
  :doc:`readability-uppercase-literal-suffix
  <clang-tidy/checks/readability-uppercase-literal-suffix>`
  added.

- The :doc:`cppcoreguidelines-narrowing-conversions
  <clang-tidy/checks/cppcoreguidelines-narrowing-conversions>` check now
  detects more narrowing conversions:
  - integer to narrower signed integer (this is compiler implementation defined),
  - integer - floating point narrowing conversions,
  - floating point - integer narrowing conversions,
  - constants with narrowing conversions (even in ternary operator).

- The :doc:`objc-property-declaration
  <clang-tidy/checks/objc-property-declaration>` check now ignores the
  `Acronyms` and `IncludeDefaultAcronyms` options.

- The :doc:`readability-redundant-smartptr-get
  <clang-tidy/checks/readability-redundant-smartptr-get>` check does not warn
  about calls inside macros anymore by default.

- The :doc:`readability-uppercase-literal-suffix
  <clang-tidy/checks/readability-uppercase-literal-suffix>` check does not warn
  about literal suffixes inside macros anymore by default.

Improvements to include-fixer
-----------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...
