========================
lld 10.0.0 Release Notes
========================

.. contents::
    :local:


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

* Relax MIPS ``jalr`` and ``jr`` instructions marked by the ``R_MIPS_JALR``
  relocation.
* For certain "undefined symbol" errors, a definition with a close spelling will be suggested.
  (`D67039 <https://reviews.llvm.org/D67039>`_)
* ``extern "C"`` is suggested if an undefined reference is mangled(unmangled) while there
  is a likely unmangled(mangled) definition.
  (`D69592 <https://reviews.llvm.org/D69592>`_ `D69650 <https://reviews.llvm.org/D69650>`_)
* New ``-z noseparate-code``, ``-z separate-code`` and ``-z separate-loadable-segments``.
  ``-z noseparate-code`` is the default, which can reduce sizes of linked binaries by up to
  3 times maxpagesize.
  (`D64903 <https://reviews.llvm.org/D64903>`_ `D67481 <https://reviews.llvm.org/D67481>`_)
* ``-z force-bti`` and ``-z pac-plt`` are added for AArch64 Branch Target Identification and Pointer Authentication.
  (`D62609 <https://reviews.llvm.org/D62609>`_)
* ``--fix-cortex-a8`` is added to fix erratum 657417.
  (`D67284 <https://reviews.llvm.org/D67284>`_)
* ``-z force-ibt`` and ``-z shstk`` are added for Intel Control-flow Enforcement Technology.
  (`D59780 <https://reviews.llvm.org/D59780>`_)
* ``PT_GNU_PROPERTY`` is added to help loaders locate the ``.note.gnu.property`` section.
  It may be used by a future Linux kernel.
  (`D70961 <https://reviews.llvm.org/D70961>`_)
* For ``--compress-debug-sections=zlib``, ``-O0`` and ``-O1`` enable compression level 1
  while ``-O2`` enables compression level 6. ``-O1`` (default) is faster than before.
  (`D70658 <https://reviews.llvm.org/D70658>`_)
* Range extension thunks with addends are implemented for AArch64, PowerPC32 and PowerPC64.
  (`D70637 <https://reviews.llvm.org/D70637>`_ `D70937 <https://reviews.llvm.org/D70937>`_
  `D73424 <https://reviews.llvm.org/D73424>`_)
* ``R_RISCV_ALIGN`` will be errored because linker relaxation for RISC-V is not supported.
  Pass ``-mno-relax`` to disable ``R_RISCV_ALIGN``.
  (`D71820 <https://reviews.llvm.org/D71820>`_)
* The ARM port will no longer insert interworking thunks for non STT_FUNC symbols.
  (`D73474 <https://reviews.llvm.org/D73474>`_)
* The quality of PowerPC32 port has been greatly improved (canonical PLT, copy
  relocations, non-preemptible IFUNC, range extension thunks with addends).
  It can link FreeBSD 13.0 userland.
* The PowerPC64 port supports non-preemptible IFUNC.
  (`D71509 <https://reviews.llvm.org/D71509>`_)
* lld creates a RO PT_LOAD and a RX PT_LOAD without a linker script.
  lld creates a unified RX PT_LOAD with a linker script.
  A future release will eliminate this difference and use a RO PT_LOAD and a RX PT_LOAD by default.
  The linker script case will require ``--no-rosegment`` to restore the current behavior.
* GNU style compressed debug sections ``.zdebug`` (obsoleted by ``SHF_COMPRESSED``)
  are supported for input files, but not for the output.
  A future release may drop ``.zdebug`` support.

Breaking changes
----------------

* ``-Ttext=$base`` (base is usually 0) is no longer supported.
  If PT_PHDR is needed, use ``--image-base=$base`` instead.
  If PT_PHDR is not needed, use a linker script with `.text 0 : { *(.text*) }` as the first
  output section description.
  See https://bugs.llvm.org/show_bug.cgi?id=44715 for more information.
  (`D67325 <https://reviews.llvm.org/D67325>`_)
* ``-Ttext-segment`` is no longer supported. Its meaning was different from GNU ld's and
  could cause subtle bugs.
  (`D70468 <https://reviews.llvm.org/D70468>`_)


MinGW Improvements
------------------

* Allow using custom .edata sections from input object files (for use
  by Wine)
  (`dadc6f248868 <https://reviews.llvm.org/rGdadc6f248868>`_)

* Don't implicitly create import libraries unless requested
  (`6540e55067e3 <https://reviews.llvm.org/rG6540e55067e3>`_)

* Support merging multiple resource object files
  (`3d3a9b3b413d <https://reviews.llvm.org/rG3d3a9b3b413d>`_)
  and properly handle the default manifest object files that GCC can pass
  (`d581dd501381 <https://reviews.llvm.org/rGd581dd501381>`_)

* Demangle itanium symbol names in warnings/error messages
  (`a66fc1c99f3e <https://reviews.llvm.org/rGa66fc1c99f3e>`_)

* Print source locations for undefined references and duplicate symbols,
  if possible
  (`1d06d48bb346 <https://reviews.llvm.org/rG1d06d48bb346>`_)
  and
  (`b38f577c015c <https://reviews.llvm.org/rGb38f577c015c>`_)

* Look for more filename patterns when resolving ``-l`` options
  (`0226c35262df <https://reviews.llvm.org/rG0226c35262df>`_)

* Don't error out on duplicate absolute symbols with the same value
  (which can happen for the default-null symbol for weak symbols)
  (`1737cc750c46 <https://reviews.llvm.org/rG1737cc750c46>`_)


WebAssembly Improvements
------------------------

* `__data_end` and `__heap_base` are no longer exported by default,
  as it's best to keep them internal when possible. They can be
  explicitly exported with `--export=__data_end` and
  `--export=__heap_base`, respectively.
* wasm-ld now elides .bss sections when the memory is not imported
