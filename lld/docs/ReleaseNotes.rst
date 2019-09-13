========================
lld 10.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 10.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release 10.0.0.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* ...

COFF Improvements
-----------------

* /linkrepro: now takes the filename of the tar archive it writes, instead
  of the name of a directory that a file called "repro.tar" is created in,
  matching the behavior of ELF lld.
* The new `/lldignoreenv` flag makes lld-link ignore environment variables
  like `%LIB%`.
* ...

MinGW Improvements
------------------

* ...

MachO Improvements
------------------

* Item 1.

WebAssembly Improvements
------------------------

* `__data_end` and `__heap_base` are no longer exported by default,
  as it's best to keep them internal when possible. They can be
  explicitly exported with `--export=__data_end` and
  `--export=__heap_base`, respectively.
