========================
lld 12.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 12.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release 12.0.0.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* ``--dependency-file`` has been added. (Similar to ``cc -M -MF``.)
  (`D82437 <https://reviews.llvm.org/D82437>`_)
* ``--error-handling-script`` has been added to allow for user-defined handlers upon
  missing libraries. (`D87758 <https://reviews.llvm.org/D87758>`_)
* ``--exclude-libs`` can now localize defined version symbols and bitcode referenced libcall symbols.
  (`D94280 <https://reviews.llvm.org/D94280>`_)
* ``--gdb-index`` now works with DWARF v5 and ``--icf={safe,all}``.
  (`D85579 <https://reviews.llvm.org/D85579>`_)
  (`D89751 <https://reviews.llvm.org/D89751>`_)
* ``--gdb-index --emit-relocs`` can now be used together.
  (`D94354 <https://reviews.llvm.org/D94354>`_)
* ``--icf={safe,all}`` conservatively no longer fold text sections with LSDA.
  Previously ICF on ``-fexceptions`` code could be unsafe.
  (`D84610 <https://reviews.llvm.org/D84610>`_)
* ``--icf={safe,all}`` can now fold two sections with relocations referencing aliased symbols.
  (`D88830 <https://reviews.llvm.org/D88830>`_)
* ``--lto-pseudo-probe-for-profiling`` has been added.
  (`D95056 <https://reviews.llvm.org/D95056>`_)
* ``--no-lto-whole-program-visibility`` has been added.
  (`D92060 <https://reviews.llvm.org/D92060>`_)
* ``--oformat-binary`` has been fixed to respect LMA.
  (`D85086 <https://reviews.llvm.org/D85086>`_)
* ``--reproduce`` includes ``--lto-sample-profile``, ``--just-symbols``, ``--call-graph-ordering-file``, ``--retain-symbols-file`` files.
* ``-r --gc-sections`` is now supported.
  (`D84131 <https://reviews.llvm.org/D84131>`_)
* A ``-u`` specified symbol will no longer change the binding to ``STB_WEAK``.
  (`D88945 <https://reviews.llvm.org/D88945>`_)
* ``--wrap`` support has been improved.
  + If ``foo`` is not referenced, there is no longer an undefined symbol ``__wrap_foo``.
  + If ``__real_foo`` is not referenced, there is no longer an undefined symbol ``foo``.
* ``SHF_LINK_ORDER`` sections can now have zero ``sh_link`` values.
* ``SHF_LINK_ORDER`` and non-``SHF_LINK_ORDER`` sections can now be mixed within an input section description.
  (`D84001 <https://reviews.llvm.org/D84001>`_)
* ``LOG2CEIL`` is now supported in linker scripts.
  (`D84054 <https://reviews.llvm.org/D84054>`_)
* ``DEFINED`` has been fixed to check whether the symbol is defined.
  (`D83758 <https://reviews.llvm.org/D83758>`_)
* An input section description may now have multiple ``SORT_*``.
  The matched sections are ordered by radix sort with the keys being ``(SORT*, --sort-section, input order)``.
  (`D91127 <https://reviews.llvm.org/D91127>`_)
* Users can now provide a GNU style linker script to convert ``.ctors`` into ``.init_array``.
  (`D91187 <https://reviews.llvm.org/D91187>`_)
* An empty output section can now be discarded even if it is assigned to a program header.
  (`D92301 <https://reviews.llvm.org/D92301>`_)
* Non-``SHF_ALLOC`` sections now have larger file offsets than ``SHF_ALLOC`` sections.
  (`D85867 <https://reviews.llvm.org/D85867>`_)
* Some symbol versioning improvements.
  + Defined ``foo@@v1`` now resolve undefined ``foo@v1`` (`D92259 <https://reviews.llvm.org/D92259>`_)
  + Undefined ``foo@v1`` now gets an error (`D92260 <https://reviews.llvm.org/D92260>`_)
* The AArch64 port now has support for ``STO_AARCH64_VARIANT_PCS`` and ``DT_AARCH64_VARIANT_PCS``.
  (`D93045 <https://reviews.llvm.org/D93045>`_)
* The AArch64 port now has support for ``R_AARCH64_LD64_GOTPAGE_LO15``.
* The PowerPC64 port now detects missing R_PPC64_TLSGD/R_PPC64_TLSLD and disables TLS relaxation.
  This allows linking with object files produced by very old IBM XL compilers.
  (`D92959 <https://reviews.llvm.org/D92959>`_)
* Many PowerPC PC-relative relocations are now supported.
* ``R_PPC_ADDR24`` and ``R_PPC64_ADDR16_HIGH`` are now supported.
* powerpcle is now supported. Tested with FreeBSD loader and freestanding.
  (`D93917 <https://reviews.llvm.org/D93917>`_)
* RISC-V: the first ``SHT_RISCV_ATTRIBUTES`` section is now retained.
  (`D86309 <https://reviews.llvm.org/D86309>`_)
* LTO pipeline now defaults to the new PM if the CMake variable ``ENABLE_EXPERIMENTAL_NEW_PASS_MANAGER`` is on.
  (`D92885 <https://reviews.llvm.org/D92885>`_)

Breaking changes
----------------

* A COMMON symbol can now cause the fetch of an archive providing a ``STB_GLOBAL`` definition.
  This behavior follows GNU ld newer than December 1999.
  If you see ``duplicate symbol`` errors with the new behavior, check out `PR49226 <https://bugs.llvm.org//show_bug.cgi?id=49226>`_.
  (`D86142 <https://reviews.llvm.org/D86142>`_)

COFF Improvements
-----------------

* Error out clearly if creating a DLL with too many exported symbols.
  (`D86701 <https://reviews.llvm.org/D86701>`_)

MinGW Improvements
------------------

* Enabled dynamicbase by default. (`D86654 <https://reviews.llvm.org/D86654>`_)

* Tolerate mismatches between COMDAT section sizes with different amount of
  padding (produced by binutils) by inspecting the aux section definition.
  (`D86659 <https://reviews.llvm.org/D86659>`_)

* Support setting the subsystem version via the subsystem argument.
  (`D88804 <https://reviews.llvm.org/D88804>`_)

* Implemented the GNU -wrap option.
  (`D89004 <https://reviews.llvm.org/D89004>`_,
  `D91689 <https://reviews.llvm.org/D91689>`_)

* Handle the ``--demangle`` and ``--no-demangle`` options.
  (`D93950 <https://reviews.llvm.org/D93950>`_)


Mach-O Improvements
------------------

We've gotten the new implementation of LLD for Mach-O to the point where it is
able to link large x86_64 programs, and we'd love to get some alpha testing on
it. The new Darwin back-end can be invoked as follows:

.. code-block::
   clang -fuse-ld=lld.darwinnew /path/to/file.c

To reach this point, we implemented numerous features, and it's easier to list
the major features we *haven't* yet completed:

* LTO support
* Stack unwinding for exceptions
* Support for arm64, arm, and i386 architectures

If you stumble upon an issue and it doesn't fall into one of these categories,
please file a bug report!


WebAssembly Improvements
------------------------

