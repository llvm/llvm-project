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

* ``ELFCOMPRESS_ZSTD`` compressed input sections are now supported.
  (`D129406 <https://reviews.llvm.org/D129406>`_)
* ``--compress-debug-sections=zstd`` is now available to compress debug
  sections with zstd (``ELFCOMPRESS_ZSTD``).
  (`D133548 <https://reviews.llvm.org/D133548>`_)
* ``--no-warnings``/``-w`` is now available to suppress warnings.
  (`D136569 <https://reviews.llvm.org/D136569>`_)
* ``DT_RISCV_VARIANT_CC`` is now produced if at least one ``R_RISCV_JUMP_SLOT``
  relocation references a symbol with the ``STO_RISCV_VARIANT_CC`` bit.
  (`D107951 <https://reviews.llvm.org/D107951>`_)
* ``--remap-inputs=`` and ``--remap-inputs-file=`` are added to remap input files.
  (`D148859 <https://reviews.llvm.org/D148859>`_)
* ``PT_RISCV_ATTRIBUTES`` is added to include the SHT_RISCV_ATTRIBUTES section.
  (`D152065 <https://reviews.llvm.org/D152065>`_)

Breaking changes
----------------

COFF Improvements
-----------------

* lld-link can now find libraries with relative paths that are relative to
  `/libpath`. Before it would only be able to find libraries relative to the
  current directory.
  I.e. ``lld-link /libpath:c:\relative\root relative\path\my.lib`` where before
  we would have to do ``lld-link /libpath:c:\relative\root\relative\path my.lib``
* lld-link learned -print-search-paths that will print all the paths where it will
  search for libraries.
* By default lld-link will now search for libraries in the toolchain directories.
  Specifically it will search:
  ``<toolchain>/lib``, ``<toolchain>/lib/clang/<version>/lib`` and
  ``<toolchain>/lib/clang/<version>/lib/windows``.

MinGW Improvements
------------------

MachO Improvements
------------------

WebAssembly Improvements
------------------------

Fixes
#####

* Arm exception index tables (.ARM.exidx sections) are now output
  correctly when they are at a non zero offset within their output
  section. (`D148033 <https://reviews.llvm.org/D148033>`_)
