===================================================
Extra Clang Tools 9.0.0 (In-Progress) Release Notes
===================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 9 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 9.0.0. Here we describe the status of the Extra Clang Tools in
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

What's New in Extra Clang Tools 9.0.0?
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

- ...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- New OpenMP module.

  For checks specific to `OpenMP <https://www.openmp.org/>`_ API.

- New :doc:`abseil-duration-addition
  <clang-tidy/checks/abseil-duration-addition>` check.

  Checks for cases where addition should be performed in the ``absl::Time``
  domain.

- New :doc:`abseil-duration-conversion-cast
  <clang-tidy/checks/abseil-duration-conversion-cast>` check.

  Checks for casts of ``absl::Duration`` conversion functions, and recommends
  the right conversion function instead.

- New :doc:`abseil-duration-unnecessary-conversion
  <clang-tidy/checks/abseil-duration-unnecessary-conversion>` check.

  Finds and fixes cases where ``absl::Duration`` values are being converted to
  numeric types and back again.

- New :doc:`abseil-time-comparison
  <clang-tidy/checks/abseil-time-comparison>` check.

  Prefer comparisons in the ``absl::Time`` domain instead of the integer
  domain.

- New :doc:`abseil-time-subtraction
  <clang-tidy/checks/abseil-time-subtraction>` check.

  Finds and fixes ``absl::Time`` subtraction expressions to do subtraction
  in the Time domain instead of the numeric domain.

- New :doc:`google-readability-avoid-underscore-in-googletest-name
  <clang-tidy/checks/google-readability-avoid-underscore-in-googletest-name>`
  check.

  Checks whether there are underscores in googletest test and test case names in
  test macros, which is prohibited by the Googletest FAQ.

- New :doc:`objc-super-self <clang-tidy/checks/objc-super-self>` check.

  Finds invocations of ``-self`` on super instances in initializers of
  subclasses of ``NSObject`` and recommends calling a superclass initializer
  instead.

- New alias :doc:`cppcoreguidelines-explicit-virtual-functions
  <clang-tidy/checks/cppcoreguidelines-explicit-virtual-functions>` to
  :doc:`modernize-use-override
  <clang-tidy/checks/modernize-use-override>` was added.

- The :doc:`bugprone-argument-comment
  <clang-tidy/checks/bugprone-argument-comment>` now supports
  `CommentBoolLiterals`, `CommentIntegerLiterals`, `CommentFloatLiterals`,
  `CommentUserDefiniedLiterals`, `CommentStringLiterals`,
  `CommentCharacterLiterals` & `CommentNullPtrs` options.

- The :doc:`bugprone-too-small-loop-variable
  <clang-tidy/checks/bugprone-too-small-loop-variable>` now supports
  `MagnitudeBitsUpperLimit` option. The default value was set to 16,
  which greatly reduces warnings related to loops which are unlikely to
  cause an actual functional bug.

- The :doc:`google-runtime-int <clang-tidy/checks/google-runtime-int>`
  check has been disabled in Objective-C++.

- The `Acronyms` and `IncludeDefaultAcronyms` options for the
  :doc:`objc-property-declaration <clang-tidy/checks/objc-property-declaration>`
  check have been removed.

- The :doc:`modernize-use-override
  <clang-tidy/checks/modernize-use-override>` now supports `OverrideSpelling`
  and `FinalSpelling` options.

- New :doc:`llvm-prefer-isa-or-dyn-cast-in-conditionals
  <clang-tidy/checks/llvm-prefer-isa-or-dyn-cast-in-conditionals>` check.

  Looks at conditionals and finds and replaces cases of ``cast<>``,
  which will assert rather than return a null pointer, and
  ``dyn_cast<>`` where the return value is not captured. Additionally,
  finds and replaces cases that match the pattern ``var &&
  isa<X>(var)``, where ``var`` is evaluated twice.

- New :doc:`modernize-use-trailing-type-return
  <clang-tidy/checks/modernize-use-trailing-type-return>` check.

  Rewrites function signatures to use a trailing return type.

Improvements to include-fixer
-----------------------------

- New :doc:`openmp-exception-escape
  <clang-tidy/checks/openmp-exception-escape>` check.

  Analyzes OpenMP Structured Blocks and checks that no exception escapes
  out of the Structured Block it was thrown in.

- New :doc:`openmp-use-default-none
  <clang-tidy/checks/openmp-use-default-none>` check.

  Finds OpenMP directives that are allowed to contain a ``default`` clause,
  but either don't specify it or the clause is specified but with the kind
  other than ``none``, and suggests to use the ``default(none)`` clause.

Improvements to clang-include-fixer
-----------------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...

Improvements to pp-trace
------------------------

- Added a new option `-callbacks` to filter preprocessor callbacks. It replaces
  the `-ignore` option.
