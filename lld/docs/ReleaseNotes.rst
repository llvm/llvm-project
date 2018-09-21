=======================
lld 8.0.0 Release Notes
=======================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 8.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release 8.0.0.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* lld now supports RISC-V. (`r339364
  <https://reviews.llvm.org/rL342746>`_)

* Default image base address has changed from 65536 to 2 MiB for i386
  and 4 MiB for AArch64 to make lld-generated executables work better
  with automatic superpage promotion. FreeBSD can promotes contiguous
  non-superpages to a superpage if they are aligned to the superpage
  size. (`r342746 <https://reviews.llvm.org/rL342746>`_)

* The following flags have been added: ``-z interpose``, ``-z global``

COFF Improvements
-----------------

* PDB GUID is set to hash of PDB contents instead to a random byte
  sequence for build reproducibility.

* The following flags have been added: ``/force:multiple``

MachO Improvements
------------------

* Item 1.
