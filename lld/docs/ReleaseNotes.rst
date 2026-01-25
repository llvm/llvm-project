.. If you want to modify sections/contents permanently, you should modify both
   ReleaseNotes.rst and ReleaseNotesTemplate.txt.

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

* Added ``--print-gc-sections=<file>`` to redirect garbage collection section
  listing to a file, avoiding contamination of stdout with other linker output.
  (`#159706 <https://github.com/llvm/llvm-project/pull/159706>`_)
* Added ``VersionNode`` lexer state for better version script parsing.
  This brings the lexer behavior closer to GNU ld.
  (`#174530 <https://github.com/llvm/llvm-project/pull/174530>`_)
* Unversioned undefined symbols now use version index 0, aligning with
  GNU ld 2.46 behavior.
  (`#168189 <https://github.com/llvm/llvm-project/pull/168189>`_)
* ``.data.rel.ro.hot`` and ``.data.rel.ro.unlikely`` are now recognized as
  RELRO sections, allowing profile-guided static data partitioning.
  (`#148920 <https://github.com/llvm/llvm-project/pull/148920>`_)
* DTLTO now supports archive members and bitcode members of thin archives.
  (`#157043 <https://github.com/llvm/llvm-project/pull/157043>`_)
* For DTLTO, ``--thinlto-remote-compiler-prepend-arg=<arg>`` has been added to
  prepend an argument to the remote compiler's command line.
  (`#162456 <https://github.com/llvm/llvm-project/pull/162456>`_)
* Balanced Partitioning (BP) section ordering now skips input sections with
  null data, and filters out section symbols.
  (`#149265 <https://github.com/llvm/llvm-project/pull/149265>`_)
  (`#151685 <https://github.com/llvm/llvm-project/pull/151685>`_)
* For AArch64, fixed a crash when using ``--fix-cortex-a53-843419`` with
  synthetic sections and improved handling when patched code is far from
  the short jump.
  (`#170495 <https://github.com/llvm/llvm-project/pull/170495>`_)
* For AArch64, added support for the ``R_AARCH64_FUNCINIT64`` dynamic
  relocation type for relocating word-sized data using the return value
  of a function.
  (`#156564 <https://github.com/llvm/llvm-project/pull/156564>`_)
* For AArch64, added support for the ``R_AARCH64_PATCHINST`` relocation type
  to support deactivation symbols.
  (`#133534 <https://github.com/llvm/llvm-project/pull/133534>`_)
* For AArch64, added support for reading AArch64 Build Attributes and
  converting them into GNU Properties.
  (`#147970 <https://github.com/llvm/llvm-project/pull/147970>`_)
* For ARM, fixed incorrect veneer generation for wraparound branches at
  the high end of the 32-bit address space branching to the low end.
  (`#165263 <https://github.com/llvm/llvm-project/pull/165263>`_)
* For LoongArch, ``-r`` now synthesizes ``R_LARCH_ALIGN`` at input section
  start to preserve alignment information.
  (`#153935 <https://github.com/llvm/llvm-project/pull/153935>`_)
* For RISC-V, added infrastructure for handling vendor-specific relocations.
  (`#159987 <https://github.com/llvm/llvm-project/pull/159987>`_)
* For RISC-V, added support for statically resolved vendor-specific relocations.
  (`#169273 <https://github.com/llvm/llvm-project/pull/169273>`_)
* For RISC-V, ``-r`` now synthesizes ``R_RISCV_ALIGN`` at input section start
  to preserve alignment information during two-stage linking.
  (`#151639 <https://github.com/llvm/llvm-project/pull/151639>`_)

Breaking changes
----------------

COFF Improvements
-----------------

* Added ``-prefetch-inputs`` to improve link times by asynchronously loading input files in RAM.
  This will dampen the effect of input file I/O latency on link times.
  However this flag can have an adverse effect when linking a large number of inputs files, or if all
  inputs do not fit in RAM at once. For those cases, linking might be a bit slower since the inputs
  will be streamed into RAM upfront, only to be evicted later by swapping.
  (`#169224 <https://github.com/llvm/llvm-project/pull/169224>`_)
* Added ``/sectionlayout:@<file>`` to specify custom output section ordering.
  (`#152779 <https://github.com/llvm/llvm-project/pull/152779>`_)
* Added ``/nodbgdirmerge`` to emit the debug directory section in ``.cvinfo``
  instead of merging it to ``.rdata``.
  (`#159235 <https://github.com/llvm/llvm-project/pull/159235>`_)
* Added ``-fat-lto-objects`` to support FatLTO. Without ``-fat-lto-objects`` or
  with ``-fat-lto-objects:no``, LLD will link LLVM FatLTO objects using the
  relocatable object file.
  (`#165529 <https://github.com/llvm/llvm-project/pull/165529>`_)
* Added ``/linkreprofullpathrsp`` to print the full path to each object
  passed to the link line to a file. This is used in particular when linking
  Arm64X binaries.
  (`#174971 <https://github.com/llvm/llvm-project/pull/174971>`_)
* Added CET flags: ``/cetcompatstrict``, ``/cetipvalidationrelaxed``,
  ``/cetdynamicapisinproc``, and ``/hotpatchcompatible``.
  (`#150761 <https://github.com/llvm/llvm-project/pull/150761>`_)
* Added support for ARM64X same-address thunks.
  (`#151255 <https://github.com/llvm/llvm-project/pull/151255>`_)
* Added more ``--time-trace`` tags for ThinLTO linking.
  (`#156471 <https://github.com/llvm/llvm-project/pull/156471>`_)
* ``/summary`` now works when ``/debug`` isn't provided.
  (`#157476 <https://github.com/llvm/llvm-project/pull/157476>`_)
* ``/summary`` now displays the size of all consumed inputs.
  (`#157284 <https://github.com/llvm/llvm-project/pull/157284>`_)
* For DTLTO, ``-thinlto-remote-compiler-prepend-arg:<arg>`` has been added to
  prepend an argument to the remote compiler's command line.
  (`#162456 <https://github.com/llvm/llvm-project/pull/162456>`_)
* Loop and SLP vectorize options are now passed to the LTO backend.
  (`#173041 <https://github.com/llvm/llvm-project/pull/173041>`_)
* Deduplicate common chunks when linking COFF files.
  (`#162553 <https://github.com/llvm/llvm-project/pull/162553>`_)
* Discard ``.llvmbc`` and ``.llvmcmd`` sections.
  (`#150897 <https://github.com/llvm/llvm-project/pull/150897>`_)
* Prevent emitting relocations for discarded weak wrapped symbols.
  (`#156214 <https://github.com/llvm/llvm-project/pull/156214>`_)

MinGW Improvements
------------------

* Added ``--fat-lto-objects`` flag.
  (`#174962 <https://github.com/llvm/llvm-project/pull/174962>`_)
* Handle ``-m mipspe`` for MIPS.
  (`#157742 <https://github.com/llvm/llvm-project/pull/157742>`_)
* Fixed implicit DLL entry point for MinGW.
  (`#171680 <https://github.com/llvm/llvm-project/pull/171680>`_)

MachO Improvements
------------------

* Added ``--read-workers=<N>`` for multi-threaded preload of input files
  into memory, significantly reducing link times for large projects.
  (`#147134 <https://github.com/llvm/llvm-project/pull/147134>`_)
* Added ``--separate-cstring-literal-sections`` to emit cstring literals
  into sections defined by their section name.
  (`#158720 <https://github.com/llvm/llvm-project/pull/158720>`_)
* Added ``--tail-merge-strings`` to enable tail merging of cstrings.
  (`#161262 <https://github.com/llvm/llvm-project/pull/161262>`_)
* Added ``--lto-emit-llvm`` command line option.
* Added ``--slop-scale`` flag for adjusting slop scale.
  (`#164295 <https://github.com/llvm/llvm-project/pull/164295>`_)
* Added support for section branch relocations, including the 1-byte form.
  (`#169062 <https://github.com/llvm/llvm-project/pull/169062>`_)
* Enabled Linker Optimization Hints pass for arm64_32.
  (`#148964 <https://github.com/llvm/llvm-project/pull/148964>`_)
* Read cstring order for non-deduped sections.
  (`#161879 <https://github.com/llvm/llvm-project/pull/161879>`_)
* Allow independent override of weak symbols aliased via ``.set``.
  (`#167825 <https://github.com/llvm/llvm-project/pull/167825>`_)
* Fixed segfault while processing malformed object file.
  (`#167025 <https://github.com/llvm/llvm-project/pull/167025>`_)
* Fixed infinite recursion when parsing corrupted export tries.
  (`#152569 <https://github.com/llvm/llvm-project/pull/152569>`_)
* Error out gracefully when offset is outside literal section.
  (`#164660 <https://github.com/llvm/llvm-project/pull/164660>`_)
* Process OSO prefix only textually in both input and output.
  (`#152063 <https://github.com/llvm/llvm-project/pull/152063>`_)

WebAssembly Improvements
------------------------

* ``--stack-first`` is now the default. Use ``--no-stack-first`` for the
  old behavior.
  (`#166998 <https://github.com/llvm/llvm-project/pull/166998>`_)
* ``--import-memory`` can now take a single name (imports from default module).
  (`#160409 <https://github.com/llvm/llvm-project/pull/160409>`_)
* ``-r`` now forces ``-Bstatic``.
  (`#108264 <https://github.com/llvm/llvm-project/pull/108264>`_)
* LTO now uses PIC reloc model with dynamic imports.
  (`#165342 <https://github.com/llvm/llvm-project/pull/165342>`_)
* Honor command line reloc model during LTO.
  (`#164838 <https://github.com/llvm/llvm-project/pull/164838>`_)
* Fixed visibility of ``__stack_pointer`` global.
  (`#161284 <https://github.com/llvm/llvm-project/pull/161284>`_)
* Fixed check for exporting mutable globals.
  (`#160787 <https://github.com/llvm/llvm-project/pull/160787>`_)
* Fixed check for implicitly exported mutable globals.
  (`#160966 <https://github.com/llvm/llvm-project/pull/160966>`_)
* Don't export deps for unused stub symbols.
  (`#173422 <https://github.com/llvm/llvm-project/pull/173422>`_)
* Fixed SEGFAULT when importing wrapped symbol.
  (`#169656 <https://github.com/llvm/llvm-project/pull/169656>`_)
* Error on unexpected relocation types in ``-pie``/``-shared`` data sections.
  (`#162117 <https://github.com/llvm/llvm-project/pull/162117>`_)

Fixes
#####
