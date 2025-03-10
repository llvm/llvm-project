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
* For AArch64, support for the ``-zgcs-report-dynamic`` option has been added. This will provide users with
  the ability to check any Dynamic Objects explicitly passed to LLD for the GNU GCS Attribute Flag. This is
  required for all files when linking with GCS enabled. Unless defined by the user, ``-zgcs-report-dynamic``
  inherits its value from the ``-zgcs-report`` option, capped at the ``warning`` level to ensure that a users
  module can still compile. This behaviour is designed to match the GNU ld Linker.

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
