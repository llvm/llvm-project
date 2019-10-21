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

* Glob pattern, which you can use in linker scripts or version scripts,
  now supports `\` and `[!...]`. Except character classes
  (e.g. `[[:digit:]]`), lld's glob pattern should be fully compatible
  with GNU now. (`r375051
  <https://github.com/llvm/llvm-project/commit/48993d5ab9413f0e5b94dfa292a233ce55b09e3e>`_)

COFF Improvements
-----------------

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
