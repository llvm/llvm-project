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

* ``-z nosectionheader`` has been implemented to omit the section header table.
  The operation is similar to ``llvm-objcopy --strip-sections``.
  (`#101286 <https://github.com/llvm/llvm-project/pull/101286>`_)
* Section ``CLASS`` syntax allows binding input section to named classes. This
  allows the linker to automatically pack the input sections into memory
  regions by automatically spilling to later class references if a region would
  overflow. This reduces the toil of manually packing regions (typical for
  embedded). It also makes full LTO feasible in such cases, since IR merging
  currently prevents the linker script from referring to input files. (TODO: PR
  Reference)

Breaking changes
----------------

COFF Improvements
-----------------

MinGW Improvements
------------------

MachO Improvements
------------------

WebAssembly Improvements
------------------------

Fixes
#####
