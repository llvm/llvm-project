.. If you want to modify sections/contents permanently, you should modify both
   ReleaseNotes.rst and ReleaseNotesTemplate.txt.

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
     `the Releases Page <https://llvm.org/releases/>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release |release|. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. For the libc++ release notes,
see `this page <https://libcxx.llvm.org/ReleaseNotes.html>`_. All LLVM releases
may be downloaded from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Potentially Breaking Changes
============================

C/C++ Language Potentially Breaking Changes
-------------------------------------------

C++ Specific Potentially Breaking Changes
-----------------------------------------

ABI Changes in This Version
---------------------------

AST Dumping Potentially Breaking Changes
----------------------------------------

Clang Frontend Potentially Breaking Changes
-------------------------------------------

Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------

What's New in Clang |release|?
==============================

C++ Language Changes
--------------------

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++17 Feature Support
^^^^^^^^^^^^^^^^^^^^^

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C Language Changes
------------------

C2y Feature Support
^^^^^^^^^^^^^^^^^^^

C23 Feature Support
^^^^^^^^^^^^^^^^^^^

Non-comprehensive list of changes in this release
-------------------------------------------------

New Compiler Flags
------------------

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
----------------------

Attribute Changes in Clang
--------------------------

Improvements to Clang's diagnostics
-----------------------------------

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

NVPTX Support
^^^^^^^^^^^^^^

X86 Support
^^^^^^^^^^^

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA Support
^^^^^^^^^^^^

AIX Support
^^^^^^^^^^^

NetBSD Support
^^^^^^^^^^^^^^

WebAssembly Support
^^^^^^^^^^^^^^^^^^^

AVR Support
^^^^^^^^^^^

DWARF Support in Clang
----------------------

Floating Point Support in Clang
-------------------------------

Fixed Point Support in Clang
----------------------------

AST Matchers
------------

clang-format
------------

libclang
--------

Code Completion
---------------

Static Analyzer
---------------

New features
^^^^^^^^^^^^

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^

Improvements
^^^^^^^^^^^^

Moved checkers
^^^^^^^^^^^^^^

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------

OpenMP Support
--------------

Improvements
^^^^^^^^^^^^

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us on the `Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_.
