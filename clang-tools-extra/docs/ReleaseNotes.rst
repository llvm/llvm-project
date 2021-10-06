====================================================
Extra Clang Tools 14.0.0 (In-Progress) Release Notes
====================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 14 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 14.0.0. Here we describe the status of the Extra Clang Tools in
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

What's New in Extra Clang Tools 14.0.0?
=======================================

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

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- Added support for globbing in `NOLINT*` expressions, to simplify suppressing
  multiple warnings in the same line.

- Added support for `NOLINTBEGIN` ... `NOLINTEND` comments to suppress
  Clang-Tidy warnings over multiple lines.

New checks
^^^^^^^^^^

- New :doc:`bugprone-suspicious-memory-comparison
  <clang-tidy/checks/bugprone-suspicious-memory-comparison>` check.

  Finds potentially incorrect calls to ``memcmp()`` based on properties of the
  arguments.

- New :doc:`cppcoreguidelines-virtual-class-destructor
  <clang-tidy/checks/cppcoreguidelines-virtual-class-destructor>` check.

  Finds virtual classes whose destructor is neither public and virtual nor
  protected and non-virtual.

- New :doc:`readability-identifier-length
  <clang-tidy/checks/readability-identifier-length>` check.

  Reports identifiers whose names are too short. Currently checks local
  variables and function parameters only.


- New :doc:`readability-data-pointer <clang-tidy/checks/readability-data-pointer>` check.

  Finds cases where code could use ``data()`` rather than the address of the
  element at index 0 in a container.

New check aliases
^^^^^^^^^^^^^^^^^

- New alias :doc:`cert-exp42-c
  <clang-tidy/checks/cert-exp42-c>` to
  :doc:`bugprone-suspicious-memory-comparison
  <clang-tidy/checks/bugprone-suspicious-memory-comparison>` was added.

- New alias :doc:`cert-flp37-c
  <clang-tidy/checks/cert-flp37-c>` to
  :doc:`bugprone-suspicious-memory-comparison
  <clang-tidy/checks/bugprone-suspicious-memory-comparison>` was added.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Removed default setting `cppcoreguidelines-explicit-virtual-functions.IgnoreDestructors = "true"`,
  to match the current state of the C++ Core Guidelines.


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

The improvements are...

Clang-tidy visual studio plugin
-------------------------------
