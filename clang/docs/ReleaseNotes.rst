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

-  ...

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

- -Wweak-template-vtables, which was deprecated in the previous release and no
  longer had any effect, has been removed.

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

Windows Support
---------------

C Language Changes in Clang
---------------------------

C++ Language Changes in Clang
-----------------------------

- ...

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^

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

Build System Changes
--------------------

AST Matchers
------------

clang-format
------------

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
