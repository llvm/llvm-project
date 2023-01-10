===========================================
Clang |release| |ReleaseNotesTitle|
===========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming Clang |version| release.
     Release notes for previous releases can be found on
     `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release |release|. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

Potentially Breaking Changes
============================
These changes are ones which we think may surprise users when upgrading to
Clang |release| because of the opportunity they pose for disruption to existing
code bases.

What's New in Clang |release|?
==============================
Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

Bug Fixes
---------
- Fix crash on invalid code when looking up a destructor in a templated class
  inside a namespace. This fixes
  `Issue 59446 <https://github.com/llvm/llvm-project/issues/59446>`_.
- Fix an issue about ``decltype`` in the members of class templates derived from
  templates with related parameters. This fixes
  `Issue 58674 <https://github.com/llvm/llvm-project/issues/58674>`_.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Non-comprehensive list of changes in this release
-------------------------------------------------

New Compiler Flags
------------------

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
-------------------------

New Pragmas in Clang
--------------------
- ...

Attribute Changes in Clang
--------------------------

Windows Support
---------------

AIX Support
-----------

C Language Changes in Clang
---------------------------

C2x Feature Support
-------------------

C++ Language Changes in Clang
-----------------------------
- Improved ``-O0`` code generation for calls to ``std::forward_like``. Similarly to
  ``std::move, std::forward`` et al. it is now treated as a compiler builtin and implemented
  directly rather than instantiating the definition from the standard library.

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^

CUDA/HIP Language Changes in Clang
----------------------------------

Objective-C Language Changes in Clang
-------------------------------------

OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------

OpenMP Support in Clang
-----------------------

...

CUDA Support in Clang
---------------------

LoongArch Support in Clang
--------------------------

RISC-V Support in Clang
-----------------------

X86 Support in Clang
--------------------

WebAssembly Support in Clang
----------------------------

DWARF Support in Clang
----------------------

Arm and AArch64 Support in Clang
--------------------------------

Floating Point Support in Clang
-------------------------------

Internal API Changes
--------------------

Build System Changes
--------------------

AST Matchers
------------

clang-format
------------

clang-extdef-mapping
--------------------

libclang
--------

Static Analyzer
---------------

.. _release-notes-sanitizers:

Sanitizers
----------

Core Analysis Improvements
==========================

- ...

New Issues Found
================

- ...

Python Binding Changes
----------------------

The following methods have been added:

-  ...

Significant Known Problems
==========================

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us on the Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_.
