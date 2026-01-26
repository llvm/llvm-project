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
* Add support for LoongArch32 (LA32R and LA32S) relocations.

Breaking changes
----------------

COFF Improvements
-----------------

* ``/fat-lto-objects`` addded to support FatLTO. Without ``/fat-lto-objects`` or with ``/fat-lto-objects:no``, LLD will link LLVM FatLTO objects using the relocatable object file.
  (`#165529 <https://github.com/llvm/llvm-project/pull/165529>`_)
* ``/linkreprofullpathrsp`` prints the full path to each object passed to the link line to a file.
  (`#174971 <https://github.com/llvm/llvm-project/pull/165449>`_)
* ``-prefetch-inputs`` can improve link times by asynchronously loading input files in RAM.
  This will dampen the effect of input file I/O latency on link times.
  However this flag can have an adverse effect when linking a large number of inputs files, or if all
  inputs do not fit in RAM at once. For those cases, linking might be a bit slower since the inputs
  will be streamed into RAM upfront, only to be evicted later by swapping.
  (`#169224 <https://github.com/llvm/llvm-project/pull/169224>`_)

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

* The ``--stack-first`` flag is now enabled by default. The old
  behavior can be enabled using ``--no-stack-first``.

Fixes
#####
