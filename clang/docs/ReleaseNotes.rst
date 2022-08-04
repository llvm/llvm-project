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
- Fixes an accepts-invalid bug in C when using a ``_Noreturn`` function
  specifier on something other than a function declaration. This fixes
  `Issue 56800 <https://github.com/llvm/llvm-project/issues/56800>`_.
- Fix `#56772 <https://github.com/llvm/llvm-project/issues/56772>`_ - invalid
  destructor names were incorrectly accepted on template classes.
- Improve compile-times with large dynamic array allocations with trivial
  constructors. This fixes
  `Issue 56774 <https://github.com/llvm/llvm-project/issues/56774>`_.
- No longer assert/miscompile when trying to make a vectorized ``_BitInt`` type
  using the ``ext_vector_type`` attribute (the ``vector_size`` attribute was
  already properly diagnosing this case).

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Clang will now correctly diagnose as ill-formed a constant expression where an
  enum without a fixed underlying type is set to a value outside the range of
  the enumeration's values. Fixes
  `Issue 50055: <https://github.com/llvm/llvm-project/issues/50055>`_.
- Clang will now check compile-time determinable string literals as format strings.
  This fixes `Issue 55805: <https://github.com/llvm/llvm-project/issues/55805>`_.
- ``-Wformat`` now recognizes ``%b`` for the ``printf``/``scanf`` family of
  functions and ``%B`` for the ``printf`` family of functions. Fixes
  `Issue 56885: <https://github.com/llvm/llvm-project/issues/56885>`_.

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

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Support capturing structured bindings in lambdas
  (`P1091R3 <https://wg21.link/p1091r3>`_ and `P1381R1 <https://wg21.link/P1381R1>`).
  This fixes issues `GH52720 <https://github.com/llvm/llvm-project/issues/52720>`_,
  `GH54300 <https://github.com/llvm/llvm-project/issues/54300>`_,
  `GH54301 <https://github.com/llvm/llvm-project/issues/54301>`_,
  and `GH49430 <https://github.com/llvm/llvm-project/issues/49430>`_.




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

- ...

X86 Support in Clang
--------------------

- Support ``-mindirect-branch-cs-prefix`` for call and jmp to indirect thunk.

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

.. _release-notes-ubsan:

Undefined Behavior Sanitizer (UBSan)
------------------------------------

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
