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

- The IgnoreImplicitCastsAndParentheses traversal mode has been removed.

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

- New ``concurrency`` module.

  Includes checks related to concurrent programming (e.g. threads, fibers,
  coroutines, etc.).

New checks
^^^^^^^^^^

- New :doc:`altera-kernel-name-restriction
  <clang-tidy/checks/altera-kernel-name-restriction>` check.

  Finds kernel files and include directives whose filename is `kernel.cl`,
  `Verilog.cl`, or `VHDL.cl`.

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

- New :doc:`concurrency-mt-unsafe <clang-tidy/checks/concurrency-mt-unsafe>`
  check.

  Finds thread-unsafe functions usage. Currently knows about POSIX and
  Glibc function sets.

- New :doc:`bugprone-signal-handler
  <clang-tidy/checks/bugprone-signal-handler>` check.

  Finds functions registered as signal handlers that call non asynchronous-safe
  functions.

- New :doc:`cert-sig30-c
  <clang-tidy/checks/cert-sig30-c>` check.

  Alias to the :doc:`bugprone-signal-handler
  <clang-tidy/checks/bugprone-signal-handler>` check.

- New :doc:`performance-no-int-to-ptr
  <clang-tidy/checks/performance-no-int-to-ptr>` check.

  Diagnoses every integer to pointer cast.

- New :doc:`readability-function-cognitive-complexity
  <clang-tidy/checks/readability-function-cognitive-complexity>` check.

  Flags functions with Cognitive Complexity metric exceeding the configured limit.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`modernize-loop-convert
  <clang-tidy/checks/modernize-loop-convert>` check.

  Now able to transform iterator loops using ``rbegin`` and ``rend`` methods.

- Improved :doc:`readability-identifier-naming
  <clang-tidy/checks/readability-identifier-naming>` check.

  Added an option `GetConfigPerFile` to support including files which use
  different naming styles.

  Now renames overridden virtual methods if the method they override has a
  style violation.
  
  Added support for specifying the style of scoped ``enum`` constants. If 
  unspecified, will fall back to the style for regular ``enum`` constants.

  Added an option `IgnoredRegexp` per identifier type to suppress identifier
  naming checks for names matching a regular expression.

- Removed `google-runtime-references` check because the rule it checks does
  not exist in the Google Style Guide anymore.

- Improved :doc:`readability-redundant-string-init
  <clang-tidy/checks/readability-redundant-string-init>` check.

  Added `std::basic_string_view` to default list of ``string``-like types.

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
