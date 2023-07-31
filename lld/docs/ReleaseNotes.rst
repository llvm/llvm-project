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

* When ``--threads=`` is not specified, the number of concurrency is now capped to 16.
  A large ``--thread=`` can harm performance, especially with some system
  malloc implementations like glibc's.
  (`D147493 <https://reviews.llvm.org/D147493>`_)
* ``--remap-inputs=`` and ``--remap-inputs-file=`` are added to remap input files.
  (`D148859 <https://reviews.llvm.org/D148859>`_)
* ``--lto=`` is now available to support ``clang -funified-lto``
  (`D123805 <https://reviews.llvm.org/D123805>`_)
* ``--lto-CGO[0-3]`` is now available to control ``CodeGenOpt::Level`` independent of the LTO optimization level.
  (`D141970 <https://reviews.llvm.org/D141970>`_)
* ``--check-dynamic-relocations=`` is now correct 32-bit targets when the addend is larger than 0x80000000.
  (`D149347 <https://reviews.llvm.org/D149347>`_)
* ``--print-memory-usage`` has been implemented for memory regions.
  (`D150644 <https://reviews.llvm.org/D150644>`_)
* ``SHF_MERGE``, ``--icf=``, and ``--build-id=fast`` have switched to 64-bit xxh3.
  (`D154813 <https://reviews.llvm.org/D154813>`_)
* Quoted output section names can now be used in linker scripts.
  (`#60496 <https://github.com/llvm/llvm-project/issues/60496>`_)
* ``MEMORY`` can now be used without a ``SECTIONS`` command.
  (`D145132 <https://reviews.llvm.org/D145132>`_)
* ``REVERSE`` can now be used in input section descriptions to reverse the order of input sections.
  (`D145381 <https://reviews.llvm.org/D145381>`_)
* Program header assignment can now be used within ``OVERLAY``. This functionality was accidentally lost in 2020.
  (`D150445 <https://reviews.llvm.org/D150445>`_)
* Operators ``^`` and ``^=`` can now be used in linker scripts.
* LoongArch is now supported.
* ``DT_AARCH64_MEMTAG_*`` dynamic tags are now supported.
  (`D143769 <https://reviews.llvm.org/D143769>`_)
* AArch32 port now supports BE-8 and BE-32 modes for big-endian.
  (`D140201 <https://reviews.llvm.org/D140201>`_)
  (`D140202 <https://reviews.llvm.org/D140202>`_)
  (`D150870 <https://reviews.llvm.org/D150870>`_)
* ``R_ARM_THM_ALU_ABS_G*`` relocations are now supported.
  (`D153407 <https://reviews.llvm.org/D153407>`_)
* ``.ARM.exidx`` sections may start at non-zero output section offset.
  (`D148033 <https://reviews.llvm.org/D148033>`_)
* Arm Cortex-M Security Extensions is now implemented.
  (`D139092 <https://reviews.llvm.org/D139092>`_)
* BTI landing pads are now added to PLT entries accessed by range extension thunks or relative vtables.
  (`D148704 <https://reviews.llvm.org/D148704>`_)
  (`D153264 <https://reviews.llvm.org/D153264>`_)
* AArch64 short range thunk has been implemented to mitigate the performance loss of a long range thunk.
  (`D148701 <https://reviews.llvm.org/D148701>`_)
* ``R_AVR_8_LO8/R_AVR_8_HI8/R_AVR_8_HLO8/R_AVR_LO8_LDI_GS/R_AVR_HI8_LDI_GS`` have been implemented.
  (`D147100 <https://reviews.llvm.org/D147100>`_)
  (`D147364 <https://reviews.llvm.org/D147364>`_)
* ``--no-power10-stubs`` now works for PowerPC64.
* ``DT_PPC64_OPT`` is now supported;
  (`D150631 <https://reviews.llvm.org/D150631>`_)
* ``PT_RISCV_ATTRIBUTES`` is added to include the SHT_RISCV_ATTRIBUTES section.
  (`D152065 <https://reviews.llvm.org/D152065>`_)
* ``R_RISCV_PLT32`` is added to support C++ relative vtables.
  (`D143115 <https://reviews.llvm.org/D143115>`_)
* RISC-V global pointer relaxation has been implemented. Specify ``--relax-gp`` to enable the linker relaxation.
  (`D143673 <https://reviews.llvm.org/D143673>`_)
* The symbol value of ``foo`` is correctly handled when ``--wrap=foo`` and RISC-V linker relaxation are used.
  (`D151768 <https://reviews.llvm.org/D151768>`_)
* x86-64 large data sections are now placed away from code sections to alleviate relocation overflow pressure.
  (`D150510 <https://reviews.llvm.org/D150510>`_)

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

* A warning is now printed if the linked module contains runtime pseudo
  relocations that are too narrow, that can end up producing runtime
  errors if loaded too far away from the referenced DLL in the address
  space. (`D154777 <https://reviews.llvm.org/D154777>`_)

MachO Improvements
------------------

WebAssembly Improvements
------------------------

Fixes
#####

* Arm exception index tables (.ARM.exidx sections) are now output
  correctly when they are at a non zero offset within their output
  section. (`D148033 <https://reviews.llvm.org/D148033>`_)
