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

* ``--fat-lto-objects`` option is added to support LLVM FatLTO.
  Without ``--fat-lto-objects``, LLD will link LLVM FatLTO objects using the
  relocatable object file. (`D146778 <https://reviews.llvm.org/D146778>`_)
* common-page-size can now be larger than the system page-size.
 (`#57618 <https://github.com/llvm/llvm-project/issues/57618>`_)

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
