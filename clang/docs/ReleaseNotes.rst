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
- Fix crash when diagnosing incorrect usage of ``_Nullable`` involving alias
  templates. This fixes
  `Issue 60344 <https://github.com/llvm/llvm-project/issues/60344>`_.
- Fix confusing warning message when ``/clang:-x`` is passed in ``clang-cl``
  driver mode and emit an error which suggests using ``/TC`` or ``/TP``
  ``clang-cl`` options instead. This fixes
  `Issue 59307 <https://github.com/llvm/llvm-project/issues/59307>`_.
- Fix crash when evaluating consteval constructor of derived class whose base
  has more than one field. This fixes
  `Issue 60166 <https://github.com/llvm/llvm-project/issues/60166>`_.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- We now generate a diagnostic for signed integer overflow due to unary minus
  in a non-constant expression context. This fixes
  `Issue 31643 <https://github.com/llvm/llvm-project/issues/31643>`_
- Clang now warns by default for C++20 and later about deprecated capture of
  ``this`` with a capture default of ``=``. This warning can be disabled with
  ``-Wno-deprecated-this-capture``.

Non-comprehensive list of changes in this release
-------------------------------------------------
- Clang now saves the address of ABI-indirect function parameters on the stack,
  improving the debug information available in programs compiled without
  optimizations.
- Clang now supports ``__builtin_nondeterministic_value`` that returns a
  nondeterministic value of the same type as the provided argument.

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

Introduced a new function attribute ``__attribute__((unsafe_buffer_usage))``
to be worn by functions containing buffer operations that could cause out of
bounds memory accesses. It emits warnings at call sites to such functions when
the flag ``-Wunsafe-buffer-usage`` is enabled.

``__declspec`` attributes can now be used together with the using keyword. Before
the attributes on ``__declspec`` was ignored, while now it will be forwarded to the
point where the alias is used.

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
- Added ``-mrvv-vector-bits=`` option to give an upper and lower bound on vector
  length. Valid values are powers of 2 between 64 and 65536. A value of 32
  should eventually be supported. We also accept "zvl" to use the Zvl*b
  extension from ``-march`` or ``-mcpu`` to the be the upper and lower bound.

X86 Support in Clang
--------------------

WebAssembly Support in Clang
----------------------------

DWARF Support in Clang
----------------------

Arm and AArch64 Support in Clang
--------------------------------

- The hard-float ABI is now available in Armv8.1-M configurations that
  have integer MVE instructions (and therefore have FP registers) but
  no scalar or vector floating point computation. Previously, trying
  to select the hard-float ABI on such a target (via
  ``-mfloat-abi=hard`` or a triple ending in ``hf``) would silently
  use the soft-float ABI instead.

- Clang builtin ``__arithmetic_fence`` and the command line option ``-fprotect-parens``
  are now enabled for AArch64.

Floating Point Support in Clang
-------------------------------
- Add ``__builtin_elementwise_log`` builtin for floating point types only.
- Add ``__builtin_elementwise_log10`` builtin for floating point types only.
- Add ``__builtin_elementwise_log2`` builtin for floating point types only.

Internal API Changes
--------------------

Build System Changes
--------------------

AST Matchers
------------

clang-format
------------

- Add ``NextLineOnly`` style to option ``PackConstructorInitializers``.
  Compared to ``NextLine`` style, ``NextLineOnly`` style will not try to
  put the initializers on the current line first, instead, it will try to
  put the initializers on the next line only.

clang-extdef-mapping
--------------------

libclang
--------

- Introduced the new function ``clang_CXXMethod_isExplicit``,
  which identifies whether a constructor or conversion function cursor
  was marked with the explicit identifier.

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
