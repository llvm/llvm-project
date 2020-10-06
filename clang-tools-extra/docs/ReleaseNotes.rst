====================================================
Extra Clang Tools 12.0.0 (In-Progress) Release Notes
====================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 12 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 12.0.0. Here we describe the status of the Extra Clang Tools in
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

What's New in Extra Clang Tools 12.0.0?
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

- Checks that allow configuring names of headers to include now support wrapping
  the include in angle brackets to create a system include. For example,
  :doc:`cppcoreguidelines-init-variables
  <clang-tidy/checks/cppcoreguidelines-init-variables>` and
  :doc:`modernize-make-unique <clang-tidy/checks/modernize-make-unique>`.

New modules
^^^^^^^^^^^

- New ``altera`` module.

  Includes checks related to OpenCL for FPGA coding guidelines, based on the
  `Altera SDK for OpenCL: Best Practices Guide
  <https://www.altera.com/en_US/pdfs/literature/hb/opencl-sdk/aocl_optimization_guide.pdf>`_.

New checks
^^^^^^^^^^

- New :doc:`altera-struct-pack-align
  <clang-tidy/checks/altera-struct-pack-align>` check.

  Finds structs that are inefficiently packed or aligned, and recommends
  packing and/or aligning of said structs as needed.

- New :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines-prefer-member-initializer>` check.

  Finds member initializations in the constructor body which can be placed into
  the initialization list instead.

- New :doc:`bugprone-misplaced-pointer-arithmetic-in-alloc
  <clang-tidy/checks/bugprone-misplaced-pointer-arithmetic-in-alloc>` check.

- New :doc:`bugprone-redundant-branch-condition
  <clang-tidy/checks/bugprone-redundant-branch-condition>` check.

  Finds condition variables in nested ``if`` statements that were also checked
  in the outer ``if`` statement and were not changed.

- New :doc:`readability-function-cognitive-complexity
  <clang-tidy/checks/readability-function-cognitive-complexity>` check.

  Flags functions with Cognitive Complexity metric exceeding the configured limit.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`readability-identifier-naming
  <clang-tidy/checks/readability-identifier-naming>` check.

  Added an option `GetConfigPerFile` to support including files which use
  different naming styles.

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
