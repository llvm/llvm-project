=======================
LLD 7.0.0 Release Notes
=======================

.. contents::
    :local:

Introduction
============

lld is a high-performance linker that supports ELF (Unix), COFF (Windows),
Mach-O (macOS), MinGW and WebAssembly. lld is command-line-compatible with GNU
linkers and Microsoft link.exe, and is significantly faster than these system
default linkers.

lld 7 for ELF and COFF are production-ready. lld/ELF can build the entire
FreeBSD/AMD64 and will be the default linker of the next version of the
operating system. lld/COFF is being used to build popular large programs such as
the Chrome web browser. Mach-O, MinGW and WebAssembly supports are still
experimental.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* Fixed a lot of long-tail compatibility issues with GNU linkers.

* Added ``-z retpolineplt`` to emit a PLT entry that doesn't contain an indirect
  jump instruction to mitigate Spectre v2 vulnerability.

* Added experimental support for `SHT_RELR sections
  <https://groups.google.com/forum/#!topic/generic-abi/bX460iggiKg>`_ to create a
  compact dynamic relocation table.

* Added support for `split stacks <https://gcc.gnu.org/wiki/SplitStacks>`_.

* Added support for address significance table (section with type
  SHT_LLVM_ADDRSIG) to improve Identical Code Folding (ICF). Combined with the
  ``-faddrsig`` compiler option added to Clang 7, lld's ``--icf=all`` can now
  safely merge functions and data to generate smaller outputs than before.

* Improved ``--gdb-index`` so that it is faster (`r336790
  <https://reviews.llvm.org/rL336790>`_) and uses less memory (`r336672
  <https://reviews.llvm.org/rL336672>`_).

* Reduced memory usage of ``--compress-debug-sections`` (`r338913
  <https://reviews.llvm.org/rL338913>`_).

* Added linker script OVERLAY support (`r335714 <https://reviews.llvm.org/rL335714>`_).

* Added ``--warn-backref`` to make it easy to identify command line option order
  that doesn't work with GNU linkers (`r329636 <https://reviews.llvm.org/rL329636>`_)

* Added ld.lld.1 man page (`r324512 <https://reviews.llvm.org/rL324512>`_).

* Added support for multi-GOT.

* Added support for MIPS position-independent executable (PIE).

* Fixed MIPS TLS GOT entries for local symbols in shared libraries.

* Fixed calculation of MIPS GP relative relocations in case of relocatable
  output.

* Added support for PPCv2 ABI.

* Removed an incomplete support of PPCv1 ABI.

* Added support for Qualcomm Hexagon ISA.

* Added the following flags: ``--apply-dynamic-relocs``, ``--check-sections``,
  ``--cref``, ``--just-symbols``, ``--keep-unique``,
  ``--no-allow-multiple-definition``, ``--no-apply-dynamic-relocs``,
  ``--no-check-sections``, ``--no-gnu-unique, ``--no-pic-executable``,
  ``--no-undefined-version``, ``--no-warn-common``, ``--pack-dyn-relocs=relr``,
  ``--pop-state``, ``--print-icf-sections``, ``--push-state``,
  ``--thinlto-index-only``, ``--thinlto-object-suffix-replace``,
  ``--thinlto-prefix-replace``, ``--warn-backref``, ``-z combreloc``, ``-z
  copyreloc``, ``-z initfirst``, ``-z keep-text-section-prefix``, ``-z lazy``,
  ``-z noexecstack``, ``-z relro``, ``-z retpolineplt``, ``-z text``

COFF Improvements
-----------------

* Improved correctness of exporting mangled stdcall symbols.

* Completed support for ARM64 relocations.

* Added support for outputting PDB debug info for MinGW targets.

* Improved compatibility of output binaries with GNU binutils objcopy/strip.

* Sped up PDB file creation.

* Changed section layout to improve compatibility with link.exe.

* Added the following flags: ``--color-diagnostics={always,never,auto}``,
  ``--no-color-diagnostics``, ``/brepro``, ``/debug:full``, ``/debug:ghash``,
  ``/guard:cf``, ``/guard:longjmp``, ``/guard:nolongjmp``, ``/integritycheck``,
  ``/order``, ``/pdbsourcepath``, ``/timestamp``
