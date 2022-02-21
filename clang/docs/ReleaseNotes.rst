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

- Clang now supports the ``-fzero-call-used-regs`` feature for x86. The purpose
  of this feature is to limit Return-Oriented Programming (ROP) exploits and
  information leakage. It works by zeroing out a selected class of registers
  before function return --- e.g., all GPRs that are used within the function.
  There is an analogous ``zero_call_used_regs`` attribute to allow for finer
  control of this feature.

Bug Fixes
------------------
- ``CXXNewExpr::getArraySize()`` previously returned a ``llvm::Optional``
  wrapping a ``nullptr`` when the ``CXXNewExpr`` did not have an array
  size expression. This was fixed and ``::getArraySize()`` will now always
  either return ``None`` or a ``llvm::Optional`` wrapping a valid ``Expr*``.
  This fixes `Issue 53742 <https://github.com/llvm/llvm-project/issues/53742>`_.

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

- Added support for parameter pack expansion in `clang::annotate`.

- The ``overloadable`` attribute can now be written in all of the syntactic
  locations a declaration attribute may appear.
  This fixes `Issue 53805 <https://github.com/llvm/llvm-project/issues/53805>`_.

Windows Support
---------------

- Add support for MSVC-compatible ``/JMC``/``/JMC-`` flag in clang-cl (supports
  X86/X64/ARM/ARM64). ``/JMC`` could only be used when ``/Zi`` or ``/Z7`` is
  turned on. With this addition, clang-cl can be used in Visual Studio for the
  JustMyCode feature. Note, you may need to manually add ``/JMC`` as additional
  compile options in the Visual Studio since it currently assumes clang-cl does not support ``/JMC``.

C Language Changes in Clang
---------------------------

C2x Feature Support
-------------------

- Implemented `WG14 N2674 The noreturn attribute <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2764.pdf>`_.
- Implemented `WG14 N2935 Make false and true first-class language features <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2935.pdf>`_.

C++ Language Changes in Clang
-----------------------------

- ...

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Diagnose consteval and constexpr issues that happen at namespace scope. This
  partially addresses `Issue 51593 <https://github.com/llvm/llvm-project/issues/51593>`_.

C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P2128R6: Multidimensional subscript operator <https://wg21.link/P2128R6>`_.
- Implemented `P0849R8: auto(x): decay-copy in the language <https://wg21.link/P0849R8>`_.

CUDA Language Changes in Clang
------------------------------

Objective-C Language Changes in Clang
-------------------------------------

OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------

OpenMP Support in Clang
-----------------------

- ``clang-nvlink-wrapper`` tool introduced to support linking of cubin files
  archived in an archive. See :doc:`ClangNvlinkWrapper`.
- ``clang-linker-wrapper`` tool introduced to support linking using a new OpenMP
  target offloading method. See :doc:`ClangLinkerWrapper`.
- Support for a new driver for OpenMP target offloading has been added as an
  opt-in feature. The new driver can be selected using ``-fopenmp-new-driver``
  with clang. Device-side LTO can also be enabled using the new driver by
  passing ``-foffload-lto=`` as well. The new driver supports the following
  features:
  - Linking AMDGPU and NVPTX offloading targets.
  - Static linking using archive files.
  - Device-side LTO.

CUDA Support in Clang
---------------------

- ...

X86 Support in Clang
--------------------

DWARF Support in Clang
----------------------

Arm and AArch64 Support in Clang
--------------------------------

Floating Point Support in Clang
-------------------------------

Internal API Changes
--------------------

- Added a new attribute flag `AcceptsExprPack` that when set allows expression
  pack expansions in the parsed arguments of the corresponding attribute.
  Additionally it introduces delaying of attribute arguments, adding common
  handling for creating attributes that cannot be fully initialized prior to
  template instantiation.

Build System Changes
--------------------

AST Matchers
------------

- Expanded ``isInline`` narrowing matcher to support c++17 inline variables.

clang-format
------------

- **Important change**: Renamed ``IndentRequires`` to ``IndentRequiresClause``
  and changed the default for all styles from ``false`` to ``true``.

- Reworked and improved handling of concepts and requires. Added the
  ``RequiresClausePosition`` option as part of that.

- Changed ``BreakBeforeConceptDeclarations`` from ``Boolean`` to an enum.

- Option ``InsertBraces`` has been added to insert optional braces after control
  statements.

libclang
--------

- ...

Static Analyzer
---------------

- ...

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
contact us via the `mailing
list <https://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
