.. If you want to modify sections/contents permanently, you should modify both
   ReleaseNotes.rst and ReleaseNotesTemplate.txt.

===========================
lld |release| Release Notes
===========================

.. contents::
    :local:

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming LLVM |release| release.
     Release notes for previous releases can be found on
     `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release |release|.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* ``--print-gc-sections=<file>`` prints garbage collection section listing to a file.
  (`#159706 <https://github.com/llvm/llvm-project/pull/159706>`_)

Breaking changes
----------------

COFF Improvements
-----------------

MinGW Improvements
------------------

MachO Improvements
------------------

* ``--separate-cstring-literal-sections`` emits cstring literal sections into sections defined by their section name.
  (`#158720 <https://github.com/llvm/llvm-project/pull/158720>`_)
* ``--tail-merge-strings`` enables tail merging of cstring literals.
  (`#161262 <https://github.com/llvm/llvm-project/pull/161262>`_)

WebAssembly Improvements
------------------------

Fixes
#####
