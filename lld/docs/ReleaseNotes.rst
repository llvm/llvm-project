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
* For AArch64, added support for ``-zgcs-report-dynamic``, enabling checks for
  GNU GCS Attribute Flags in Dynamic Objects when GCS is enabled. Inherits value
  from ``-zgcs-report`` (capped at ``warning`` level) unless user-defined,
  ensuring compatibility with GNU ld linker.
* Added ``--xosegment`` and ``--no-xosegment`` flags to control whether to place
  XO and RX sections in the same segment. The default value is ``--no-xosegment``.

Breaking changes
----------------
* XO and RX sections are now allowed to be placed in the same segment by default.
  Pass ``--xosegment`` to lld in order to get the old behavior back.

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
