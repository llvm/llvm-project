===================================================
Extra Clang Tools 7.0.0 (In-Progress) Release Notes
===================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 7 release.
   Release notes for previous releases can be found on
   `the Download Page <http://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 7.0.0. Here we describe the status of the Extra Clang Tools in
some detail, including major improvements from the previous release and new
feature work. All LLVM releases may be downloaded from the `LLVM releases web
site <http://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please see the `Clang Web Site <http://clang.llvm.org>`_ or
the `LLVM Web Site <http://llvm.org>`_.

Note that if you are reading this file from a Subversion checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <http://llvm.org/releases/>`_.

What's New in Extra Clang Tools 7.0.0?
======================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

...

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- New `bugprone-throw-keyword-missing
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-throw-keyword-missing.html>`_ check

  Diagnoses when a temporary object that appears to be an exception is
  constructed but not thrown.

- New `cppcoreguidelines-avoid-goto
  <http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-avoid-goto.html>`_ check

  The usage of ``goto`` for control flow is error prone and should be replaced
  with looping constructs. Every backward jump is rejected. Forward jumps are
  only allowed in nested loops.

- New `fuchsia-multiple-inheritance
  <http://clang.llvm.org/extra/clang-tidy/checks/fuchsia-multiple-inheritance.html>`_ check

  Warns if a class inherits from multiple classes that are not pure virtual.

- New `fuchsia-statically-constructed-objects
  <http://clang.llvm.org/extra/clang-tidy/checks/fuchsia-statically-constructed-objects.html>`_ check

  Warns if global, non-trivial objects with static storage are constructed,
  unless the object is statically initialized with a ``constexpr`` constructor
  or has no explicit constructor.
  
- New `fuchsia-trailing-return
  <http://clang.llvm.org/extra/clang-tidy/checks/fuchsia-trailing-return.html>`_ check

  Functions that have trailing returns are disallowed, except for those 
  using ``decltype`` specifiers and lambda with otherwise unutterable 
  return types.

- New `modernize-use-uncaught-exceptions
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-uncaught-exceptions.html>`_ check

  Finds and replaces deprecated uses of ``std::uncaught_exception`` to
  ``std::uncaught_exceptions``.

- New `readability-simd-intrinsics
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-simd-intrinsics.html>`_ check

  Warns if SIMD intrinsics are used which can be replaced by
  ``std::experimental::simd`` operations.

- New alias `hicpp-avoid-goto
  <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-avoid-goto.html>`_ to
  `cppcoreguidelines-avoid-goto <http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-avoid-goto.html>`_
  added.

- The 'misc-forwarding-reference-overload' check was renamed to `bugprone-forwarding-reference-overload
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-forwarding-reference-overload.html>`_

- The 'misc-incorrect-roundings' check was renamed to `bugprone-incorrect-roundings
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-incorrect-roundings.html>`_

- The 'misc-lambda-function-name' check was renamed to `bugprone-lambda-function-name
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-lambda-function-name.html>`_

- The 'misc-macro-repeated-side-effects' check was renamed to `bugprone-macro-repeated-side-effects
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-macro-repeated-side-effects.html>`_

- The 'misc-misplaced-widening-cast' check was renamed to `bugprone-misplaced-widening-cast
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-misplaced-widening-cast.html>`_

- The 'misc-string-compare' check was renamed to `readability-string-compare
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-string-compare.html>`_

- The 'misc-string-integer-assignment' check was renamed to `bugprone-string-integer-assignment
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-string-integer-assignment.html>`_

- The 'misc-string-literal-with-embedded-nul' check was renamed to `bugprone-string-literal-with-embedded-nul
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-string-literal-with-embedded-nul.html>`_

- The 'misc-suspicious-enum-usage' check was renamed to `bugprone-suspicious-enum-usage
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-suspicious-enum-usage.html>`_

- The 'misc-suspicious-missing-comma' check was renamed to `bugprone-suspicious-missing-comma
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-suspicious-missing-comma.html>`_

- The 'misc-suspicious-semicolon' check was renamed to `bugprone-suspicious-semicolon
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-suspicious-semicolon.html>`_

- The 'misc-suspicious-string-compare' check was renamed to `bugprone-suspicious-string-compare
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-suspicious-string-compare.html>`_

- The 'misc-swapped-arguments' check was renamed to `bugprone-swapped-arguments
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-swapped-arguments.html>`_

- The 'misc-undelegated-constructor' check was renamed to `bugprone-undelegated-constructor
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-undelegated-constructor.html>`_

Improvements to include-fixer
-----------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...
