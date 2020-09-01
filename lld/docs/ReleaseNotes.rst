========================
lld 11.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 11.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release 11.0.0.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* ``--lto-emit-asm`` is added to emit assembly output for debugging purposes.
  (`D77231 <https://reviews.llvm.org/D77231>`_)
* ``--lto-whole-program-visibility`` is added to specify that classes have hidden LTO visibility in LTO and ThinLTO links of source files compiled with ``-fwhole-program-vtables``. See `LTOVisibility <https://clang.llvm.org/docs/LTOVisibility.html>`_ for details.
  (`D71913 <https://reviews.llvm.org/D71913>`_)
* ``--print-archive-stats=`` is added to print the number of members and the number of fetched members for each archive.
  The feature is similar to GNU gold's ``--print-symbol-counts=``.
  (`D78983 <https://reviews.llvm.org/D78983>`_)
* ``--shuffle-sections=`` is added to introduce randomization in the output to help reduce measurement bias and detect static initialization order fiasco.
  (`D74791 <https://reviews.llvm.org/D74791>`_)
  (`D74887 <https://reviews.llvm.org/D74887>`_)
* ``--time-trace`` is added. It records a time trace file that can be viewed in
  chrome://tracing. The file can be specified with ``--time-trace-file``.
  Trace granularity can be specified with ``--time-trace-granularity``.
  (`D71060 <https://reviews.llvm.org/D71060>`_)
* ``--thinlto-single-module`` is added to compile a subset of modules in ThinLTO for debugging purposes.
  (`D80406 <https://reviews.llvm.org/D80406>`_)
* ``--unique`` is added to create separate output sections for orphan sections.
  (`D75536 <https://reviews.llvm.org/D75536>`_)
* ``--warn-backrefs`` has been improved to emulate GNU ld's archive semantics.
  If a link passes with warnings from ``--warn-backrefs``, it almost assuredly
  means that the link will fail with GNU ld, or the symbol will get different
  resolutions in GNU ld and LLD. ``--warn-backrefs-exclude=`` is added to
  exclude known issues.
  (`D77522 <https://reviews.llvm.org/D77522>`_)
  (`D77630 <https://reviews.llvm.org/D77630>`_)
  (`D77512 <https://reviews.llvm.org/D77512>`_)
* ``--no-relax`` is accepted but ignored. The Linux kernel's RISC-V port uses this option.
  (`D81359 <https://reviews.llvm.org/D81359>`_)
* ``--rosegment`` (default) is added to complement ``--no-rosegment``.
  GNU gold from 2.35 onwards support both options.
* ``--threads=N`` is added. The default uses all threads.
  (`D76885 <https://reviews.llvm.org/D76885>`_)
* ``--wrap`` has better compatibility with GNU ld.
* ``-z dead-reloc-in-nonalloc=<section_glob>=<value>`` is added to resolve an absolute relocation
  referencing a discarded symbol.
  (`D83264 <https://reviews.llvm.org/D83264>`_)
* Changed tombstone values to (``.debug_ranges``/``.debug_loc``) 1 and (other ``.debug_*``) 0.
  A tombstone value is the computed value of a relocation referencing a discarded symbol (``--gc-sections``, ICF or ``/DISCARD/``).
  (`D84825 <https://reviews.llvm.org/D84825>`_)
  In the future many .debug_* may switch to 0xffffffff/0xffffffffffffffff as the tombstone value.
* ``-z keep-text-section-prefix`` moves ``.text.unknown.*`` input sections to ``.text.unknown``.
* ``-z rel`` and ``-z rela`` are added to select the REL/RELA format for dynamic relocations.
  The default is target specific and typically matches the form used in relocatable objects.
* ``-z start-stop-visibility={default,protected,internal,hidden}`` is added.
  GNU ld/gold from 2.35 onwards support this option.
  (`D55682 <https://reviews.llvm.org/D55682>`_)
* When ``-r`` or ``--emit-relocs`` is specified, the GNU ld compatible
  ``--discard-all`` and ``--discard-locals`` semantics are implemented.
  (`D77807 <https://reviews.llvm.org/D77807>`_)
* ``--emit-relocs --strip-debug`` can now be used together.
  (`D74375 <https://reviews.llvm.org/D74375>`_)
* ``--gdb-index`` supports DWARF v5.
  (`D79061 <https://reviews.llvm.org/D79061>`_)
  (`D85579 <https://reviews.llvm.org/D85579>`_)
* ``-r`` allows SHT_X86_64_UNWIND to be merged into SHT_PROGBITS.
  This allows clang/GCC produced object files to be mixed together.
  (`D85785 <https://reviews.llvm.org/D85785>`_)
* Better linker script support related to output section alignments and LMA regions.
  (`D74286 <https://reviews.llvm.org/D75724>`_)
  (`D74297 <https://reviews.llvm.org/D75724>`_)
  (`D75724 <https://reviews.llvm.org/D75724>`_)
  (`D81986 <https://reviews.llvm.org/D81986>`_)
* In a input section description, the filename can be specified in double quotes.
  ``archive:file`` syntax is added.
  (`D72517 <https://reviews.llvm.org/D72517>`_)
  (`D75100 <https://reviews.llvm.org/D75100>`_)
* Linker script specified empty ``(.init|.preinit|.fini)_array`` are allowed with RELRO.
  (`D76915 <https://reviews.llvm.org/D76915>`_)
* ``INSERT AFTER`` and ``INSERT BEFORE`` work for orphan sections now.
  (`D74375 <https://reviews.llvm.org/D74375>`_)
* ``INPUT_SECTION_FLAGS`` is supported in linker scripts.
  (`D72745 <https://reviews.llvm.org/D72745>`_)
* ``DF_1_PIE`` is set for position-independent executables.
  (`D80872 <https://reviews.llvm.org/D80872>`_)
* For a symbol assignment ``alias = aliasee;``, ``alias`` inherits the ``aliasee``'s symbol type.
  (`D86263 <https://reviews.llvm.org/D86263>`_)
* ``SHT_GNU_verneed`` in shared objects are parsed, and versioned undefined symbols in shared objects are respected.
  (`D80059 <https://reviews.llvm.org/D80059>`_)
* SHF_LINK_ORDER and non-SHF_LINK_ORDER sections can be mixed along as the SHF_LINK_ORDER components are contiguous.
  (`D77007 <https://reviews.llvm.org/D77007>`_)
* An out-of-range relocation diagnostic mentions the referenced symbol now.
  (`D73518 <https://reviews.llvm.org/D73518>`_)
* AArch64: ``R_AARCH64_PLT32`` is supported.
  (`D81184 <https://reviews.llvm.org/D81184>`_)
* ARM: SBREL type relocations are supported.
  (`D74375 <https://reviews.llvm.org/D74375>`_)
* ARM: ``R_ARM_ALU_PC_G0``, ``R_ARM_LDR_PC_G0``, ``R_ARM_THUMB_PC8`` and ``R_ARM_THUMB__PC12`` are supported.
  (`D75349 <https://reviews.llvm.org/D75349>`_)
  (`D77200 <https://reviews.llvm.org/D77200>`_)
* ARM: various improvements to .ARM.exidx: ``/DISCARD/`` support for a subset, out-of-range handling, support for non monotonic section order.
  (`PR44824 <https://llvm.org/PR44824>`_)
* AVR: many relocation types are supported.
  (`D78741 <https://reviews.llvm.org/D78741>`_)
* Hexagon: General Dynamic and some other relocation types are supported.
* PPC: Canonical PLT and range extension thunks with addends are supported.
  (`D73399 <https://reviews.llvm.org/D73399>`_)
  (`D73424 <https://reviews.llvm.org/D73424>`_)
  (`D75394 <https://reviews.llvm.org/D75394>`_)
* PPC and PPC64: copy relocations.
  (`D73255 <https://reviews.llvm.org/D73255>`_)
* PPC64: ``_savegpr[01]_{14..31}`` and ``_restgpr[01]_{14..31}`` can be synthesized.
  (`D79977 <https://reviews.llvm.org/D79977>`_)
* PPC64: ``R_PPC64_GOT_PCREL34`` and ``R_PPC64_REL24_NOTOC`` are supported. r2 save stub is supported.
  (`D81948 <https://reviews.llvm.org/D81948>`_)
  (`D82950 <https://reviews.llvm.org/D82950>`_)
  (`D82816 <https://reviews.llvm.org/D82816>`_)
* RISC-V: ``R_RISCV_IRELATIVE`` is supported.
  (`D74022 <https://reviews.llvm.org/D74022>`_)
* RISC-V: ``R_RISCV_ALIGN`` is errored because GNU ld style linker relaxation is not supported.
  (`D71820 <https://reviews.llvm.org/D71820>`_)
* SPARCv9: more relocation types are supported.
  (`D77672 <https://reviews.llvm.org/D77672>`_)

Breaking changes
----------------

* One-dash form of some long option (``--thinlto-*``, ``--lto-*``, ``--shuffle-sections=``)
  are no longer supported.
  (`D79371 <https://reviews.llvm.org/D79371>`_)
* ``--export-dynamic-symbol`` no longer implies ``-u``.
  The new behavior matches GNU ld from binutils 2.35 onwards.
  (`D80487 <https://reviews.llvm.org/D80487>`_)
* ARM: the default max page size was increased from 4096 to 65536.
  This increases compatibility with systems where a non standard page
  size was configured. This also is inline with GNU ld defaults.
  (`D77330 <https://reviews.llvm.org/D77330>`_)
* ARM: for non-STT_FUNC symbols, Thumb interworking thunks are not added and BL/BLX are not substituted.
  (`D73474 <https://reviews.llvm.org/D73474>`_)
  (`D73542 <https://reviews.llvm.org/D73542>`_)
* AArch64: ``--force-bti`` is renamed to ``-z force-bti`. ``--pac-plt`` is renamed to ``-z pac-plt``.
  This change is compatibile with GNU ld.
* A readonly ``PT_LOAD`` is created in the presence of a ``SECTIONS`` command.
  The new behavior is consistent with the longstanding behavior in the absence of a SECTIONS command.
* Orphan section names like ``.rodata.foo`` and ``.text.foo`` are not grouped into ``.rodata`` and ``.text`` in the presence of a ``SECTIONS`` command.
  The new behavior matches GNU ld.
  (`D75225 <https://reviews.llvm.org/D75225>`_)
* ``--no-threads`` is removed. Use ``--threads=1`` instead. ``--threads`` (no-op) is removed.

COFF Improvements
-----------------

* Fixed exporting symbols whose names contain a period (``.``), which was
  a regression in lld 7.

MinGW Improvements
------------------

* Implemented new options for disabling auto import and runtime pseudo
  relocations (``--disable-auto-import`` and
  ``--disable-runtime-pseudo-reloc``), the ``--no-seh`` flag and options
  for selecting file and section alignment (``--file-alignment`` and
  ``--section-alignment``).

MachO Improvements
------------------

* Item 1.

WebAssembly Improvements
------------------------

