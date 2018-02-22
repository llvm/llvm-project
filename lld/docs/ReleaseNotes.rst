=======================
LLD 6.0.0 Release Notes
=======================

.. contents::
    :local:

Introduction
============

This document contains the release notes for the LLD linker, release 6.0.0.
Here we describe the status of LLD, including major improvements
from the previous release. All LLD releases may be downloaded
from the `LLVM releases web site <http://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* MIPS port now generates all output dynamic relocations
  using Elf_Rel format only.

* Added handling of the R_MIPS_26 relocation in case of N32 / N64 ABIs
  and generating proper PLT entries.

COFF Improvements
-----------------

* A GNU ld style frontend for the COFF linker has been added for MinGW.
  In MinGW environments, the linker is invoked with GNU ld style parameters;
  which LLD previously only supported when used as an ELF linker. When
  a PE/COFF target is chosen, those parameters are rewritten into the
  lld-link style parameters and the COFF linker is invoked instead.

* Initial support for the ARM64 architecture has been added.

* New ``--version`` flag.

* Significantly improved support for writing PDB Files.

* New ``--rsp-quoting`` flag, like ``clang-cl``.

* ``/manifestuac:no`` no longer incorrectly disables ``/manifestdependency:``.

* Only write ``.manifest`` files if ``/manifest`` is passed.

MachO Improvements
------------------

* Item 1.
