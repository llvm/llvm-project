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

* New ``elf32btsmipn32_fbsd`` and ``elf32ltsmipn32_fbsd`` emulations
  are supported.

* Relax MIPS ``jalr``and ``jr`` instructions marked by the ``R_MIPS_JALR``
  relocation.

* Reduced size of linked MIPS binaries.

COFF Improvements
-----------------

* ...

MinGW Improvements
------------------

* Allow using custom .edata sections from input object files (for use
  by Wine)
  (`dadc6f248868 <https://reviews.llvm.org/rGdadc6f248868>`)

* Don't implicitly create import libraries unless requested
  (`6540e55067e3 <https://reviews.llvm.org/rG6540e55067e3>`)

* Support merging multiple resource object files
  (`3d3a9b3b413d <https://reviews.llvm.org/rG3d3a9b3b413d>`)
  and properly handle the default manifest object files that GCC can pass
  (`d581dd501381 <https://reviews.llvm.org/rGd581dd501381>`)

* Demangle itanium symbol names in warnings/error messages
  (`a66fc1c99f3e <https://reviews.llvm.org/rGa66fc1c99f3e>`)

* Print source locations for undefined references and duplicate symbols,
  if possible
  (`1d06d48bb346 <https://reviews.llvm.org/rG1d06d48bb346>`)
  and
  (`b38f577c015c <https://reviews.llvm.org/rGb38f577c015c>`)

* Look for more filename patterns when resolving ``-l`` options
  (`0226c35262df <https://reviews.llvm.org/rG0226c35262df>`)

* Don't error out on duplicate absolute symbols with the same value
  (which can happen for the default-null symbol for weak symbols)
  (`1737cc750c46 <https://reviews.llvm.org/rG1737cc750c46>`)

MachO Improvements
------------------

* Item 1.

WebAssembly Improvements
------------------------

* `__data_end` and `__heap_base` are no longer exported by default,
  as it's best to keep them internal when possible. They can be
  explicitly exported with `--export=__data_end` and
  `--export=__heap_base`, respectively.
* wasm-ld now elides .bss sections when the memory is not imported
