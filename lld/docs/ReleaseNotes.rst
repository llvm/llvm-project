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

* ``-z nosectionheader`` has been implemented to omit the section header table.
  The operation is similar to ``llvm-objcopy --strip-sections``.
  (`#101286 <https://github.com/llvm/llvm-project/pull/101286>`_)
* ``--randomize-section-padding=<seed>`` is introduced to insert random padding
  between input sections and at the start of each segment. This can be used to
  control measurement bias in A/B experiments.
  (`#117653 <https://github.com/llvm/llvm-project/pull/117653>`_)
* The reproduce tarball created with ``--reproduce=`` now excludes directories
  specified in the ``--dependency-file`` argument (used by Ninja). This
  resolves an error where non-existent directories could cause issues when
  invoking ``ld.lld @response.txt``.
* ``--symbol-ordering-file=`` and call graph profile can now be used together.
* When ``--call-graph-ordering-file=`` is specified, ``.llvm.call-graph-profile``
  sections in relocatable files are no longer used.
* ``--lto-basic-block-sections=labels`` is deprecated in favor of
  ``--lto-basic-block-address-map``.
  (`#110697 <https://github.com/llvm/llvm-project/pull/110697>`_)
* In non-relocatable links, a ``.note.GNU-stack`` section with the
  ``SHF_EXECINSTR`` flag is now rejected unless ``-z execstack`` is specified.
  (`#124068 <https://github.com/llvm/llvm-project/pull/124068>`_)
* In relocatable links, the ``sh_entsize`` member of a ``SHF_MERGE`` section
  with relocations is now respected in the output.
* Quoted names can now be used in output section phdr, memory region names,
  ``OVERLAY``, the LHS of ``--defsym``, and ``INSERT AFTER``.
* Section ``CLASS`` linker script syntax binds input sections to named classes,
  which are referenced later one or more times. This provides access to the
  automatic spilling mechanism of `--enable-non-contiguous-regions` without
  globally changing the semantics of section matching. It also independently
  increases the expressive power of linker scripts.
  (`#95323 <https://github.com/llvm/llvm-project/pull/95323>`_)
* ``INCLUDE`` cycle detection has been fixed. A linker script can now be
  included twice.
* The ``archivename:`` syntax when matching input sections is now supported.
  (`#119293 <https://github.com/llvm/llvm-project/pull/119293>`_)
* To support Arm v6-M, short thunks using B.w are no longer generated.
  (`#118111 <https://github.com/llvm/llvm-project/pull/118111>`_)
* For AArch64, BTI-aware long branch thunks can now be created to a destination
  function without a BTI instruction.
  (`#108989 <https://github.com/llvm/llvm-project/pull/108989>`_)
  (`#116402 <https://github.com/llvm/llvm-project/pull/116402>`_)
* Relocations related to GOT and TLSDESC for the AArch64 Pointer Authentication ABI
  are now supported.
* Supported relocation types for x86-64 target:
  * ``R_X86_64_CODE_4_GOTPCRELX`` (`#109783 <https://github.com/llvm/llvm-project/pull/109783>`_) (`#116737 <https://github.com/llvm/llvm-project/pull/116737>`_)
  * ``R_X86_64_CODE_4_GOTTPOFF`` (`#116634 <https://github.com/llvm/llvm-project/pull/116634>`_)
  * ``R_X86_64_CODE_4_GOTPC32_TLSDESC`` (`#116909 <https://github.com/llvm/llvm-project/pull/116909>`_)
  * ``R_X86_64_CODE_6_GOTTPOFF``  (`#117675 <https://github.com/llvm/llvm-project/pull/117675>`_)
* Supported relocation types for LoongArch target: ``R_LARCH_TLS_{LD,GD,DESC}_PCREL20_S2``.
  (`#100105 <https://github.com/llvm/llvm-project/pull/100105>`_)

Breaking changes
----------------

* Removed support for the (deprecated) `R_RISCV_RVC_LUI` relocation. This
  was a binutils-internal relocation used during relaxation, and was not
  emitted by compilers/assemblers.

COFF Improvements
-----------------
* ``/includeglob`` has been implemented to match the behavior of ``--undefined-glob`` available for ELF.
* ``/lldsavetemps`` allows saving select intermediate LTO compilation results (e.g. resolution, preopt, promote, internalize, import, opt, precodegen, prelink, combinedindex).
* ``/machine:arm64ec`` support completed, enabling the linking of ARM64EC images.
* COFF weak anti-dependency alias symbols are now supported.

MinGW Improvements
------------------
* ``--undefined-glob`` is now supported by translating into the ``/includeglob`` flag.

MachO Improvements
------------------

WebAssembly Improvements
------------------------

Fixes
#####
