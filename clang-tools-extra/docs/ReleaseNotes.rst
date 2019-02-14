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

- New :doc:`google-readability-avoid-underscore-in-googletest-name
  <clang-tidy/checks/google-readability-avoid-underscore-in-googletest-name>`
  check.

  Checks whether there are underscores in googletest test and test case names in
  test macros, which is prohibited by the Googletest FAQ.

- The :doc:`bugprone-argument-comment
  <clang-tidy/checks/bugprone-argument-comment>` now supports
  `CommentBoolLiterals`, `CommentIntegerLiterals`,  `CommentFloatLiterals`,
  `CommentUserDefiniedLiterals`, `CommentStringLiterals`,
  `CommentCharacterLiterals` & `CommentNullPtrs` options.

Improvements to include-fixer
-----------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...
