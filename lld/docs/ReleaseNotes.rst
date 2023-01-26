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
* ``--no-undefined-version`` is now the default; symbols named in version
  scripts that have no matching symbol in the output will be reported. Use
  ``--undefined-version`` to revert to the old behavior.
* The output ``SHT_RISCV_ATTRIBUTES`` section now merges all input components
  instead of picking the first input component.
  (`D138550 <https://reviews.llvm.org/D138550>`_)

Breaking changes
----------------

COFF Improvements
-----------------

* The linker command line entry in ``S_ENVBLOCK`` of the PDB is now stripped
  from input files, to align with MSVC behavior.
  (`D137723 <https://reviews.llvm.org/D137723>`_)
* Switched from SHA1 to BLAKE3 for PDB type hashing / ``-gcodeview-ghash``
  (`D137101 <https://reviews.llvm.org/D137101>`_)
* Improvements to the PCH.OBJ files handling. Now LLD behaves the same as MSVC
  link.exe when merging PCH.OBJ files that don't have the same signature.
  (`D136762 <https://reviews.llvm.org/D136762>`_)

MinGW Improvements
------------------

* The lld-specific options ``--guard-cf``, ``--no-guard-cf``,
  ``--guard-longjmp`` and ``--no-guard-longjmp`` has been added to allow
  enabling Control Flow Guard and long jump hardening. These options are
  disabled by default, but enabling ``--guard-cf`` will also enable
  ``--guard-longjmp`` unless ``--no-guard-longjmp`` is also specified.
  ``--guard-longjmp`` depends on ``--guard-cf`` and cannot be used by itself.
  Note that these features require the ``_load_config_used`` symbol to contain
  the load config directory and be filled with the required symbols.
  (`D132808 <https://reviews.llvm.org/D132808>`_)

MachO Improvements
------------------

WebAssembly Improvements
------------------------

