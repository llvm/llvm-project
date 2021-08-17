========================
lld 13.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 13.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release 13.0.0.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* ``-z start-stop-gc`` is now supported and becomes the default.
  (`D96914 <https://reviews.llvm.org/D96914>`_)
  (`rG6d2d3bd0 <https://reviews.llvm.org/rG6d2d3bd0a61f5fc7fd9f61f48bc30e9ca77cc619>`_)
* ``--shuffle-sections=<seed>`` has been changed to ``--shuffle-sections=<section-glob>=<seed>``.
  If seed is -1, the matched input sections are reversed.
  (`D98445 <https://reviews.llvm.org/D98445>`_)
  (`D98679 <https://reviews.llvm.org/D98679>`_)
* ``-Bsymbolic -Bsymbolic-functions`` has been changed to behave the same as ``-Bsymbolic-functions``. This matches GNU ld.
  (`D102461 <https://reviews.llvm.org/D102461>`_)
* ``-Bno-symbolic`` has been added.
  (`D102461 <https://reviews.llvm.org/D102461>`_)
* A new linker script command ``OVERWRITE_SECTIONS`` has been added.
  (`D103303 <https://reviews.llvm.org/D103303>`_)
* ``-Bsymbolic-non-weak-functions`` has been added as a ``STB_GLOBAL`` subset of ``-Bsymbolic-functions``.
  (`D102570 <https://reviews.llvm.org/D102570>`_)
* ``--no-allow-shlib-undefined`` has been improved to catch more cases.
  (`D101996 <https://reviews.llvm.org/D101996>`_)
* ``__rela_iplt_start`` is no longer defined for -pie/-shared.
  This makes GCC/Clang ``-static-pie`` built executables work.
  (`rG8cb78e99 <https://reviews.llvm.org/rf8cb78e99aae9aa3f89f7bfe667db2c5b767f21f>`_)
* IRELATIVE/TLSDESC relocations now support ``-z rel``.
  (`D100544 <https://reviews.llvm.org/D100544>`_)
* Section groups with a zero flag are now supported.
  This is used by ``comdat nodeduplicate`` in LLVM IR.
  (`D96636 <https://reviews.llvm.org/D96636>`_)
  (`D106228 <https://reviews.llvm.org/D106228>`_)
* Defined symbols are now resolved before undefined symbols to stabilize the bheavior of archive member extraction.
  (`D95985 <https://reviews.llvm.org/D95985>`_)
* ``STB_WEAK`` symbols are now preferred over COMMON symbols as a fix to a ``--fortran-common`` regression.
  (`D105945 <https://reviews.llvm.org/D105945>`_)
* Absolute relocations referencing undef weak now produce dynamic relocations for -pie, matching GOT-generating relocations.
  (`D105164 <https://reviews.llvm.org/D105164>`_)
* Exported symbols are now communicated to the LTO library so as to make LTO
  based whole program devirtualization (``-flto=thin -fwhole-program-vtables``)
  work with shared objects.
  (`D91583 <https://reviews.llvm.org/D91583>`_)
* Whole program devirtualization now respects ``local:`` version nodes in a version script.
  (`D98220 <https://reviews.llvm.org/D98220>`_)
  (`D98686 <https://reviews.llvm.org/D98686>`_)
* ``local:`` version nodes in a version script now apply to non-default version symbols.
  (`D107234 <https://reviews.llvm.org/D107234>`_)
* If an object file defines both ``foo`` and ``foo@v1``, now only ``foo@v1`` will be in the output.
  (`D107235 <https://reviews.llvm.org/D107235>`_)
* Copy relocations on non-default version symbols are now supported.
  (`D107535 <https://reviews.llvm.org/D107535>`_)

Linker script changes:

* ``.``, ``$``, and double quotes can now be used in symbol names in expressions.
  (`D98306 <https://reviews.llvm.org/D98306>`_)
  (`rGe7a7ad13 <https://reviews.llvm.org/rGe7a7ad134fe182aad190cb3ebc441164470e92f5>`_)
* Fixed value of ``.`` in the output section description of ``.tbss``.
  (`D107288 <https://reviews.llvm.org/D107288>`_)
* ``NOLOAD`` sections can now be placed in a ``PT_LOAD`` program header.
  (`D103815 <https://reviews.llvm.org/D103815>`_)
* ``OUTPUT_FORMAT(default, big, little)`` now consults ``-EL`` and ``-EB``.
  (`D96214 <https://reviews.llvm.org/D96214>`_)
* The ``OVERWRITE_SECTIONS`` command has been added.
  (`D103303 <https://reviews.llvm.org/D103303>`_)
* The section order within an ``INSERT AFTER`` command is now preserved.
  (`D105158 <https://reviews.llvm.org/D105158>`_)

Architecture specific changes:

* aarch64_be is now supported.
  (`D96188 <https://reviews.llvm.org/D96188>`_)
* The AMDGPU port now supports ``--amdhsa-code-object-version=4`` object files;
  (`D95811 <https://reviews.llvm.org/D95811>`_)
* The ARM port now accounts for PC biases in range extension thunk creation.
  (`D97550 <https://reviews.llvm.org/D97550>`_)
* The AVR port now computes ``e_flags``.
  (`D99754 <https://reviews.llvm.org/D99754>`_)
* The Mips port now omits unneeded dynamic relocations for PIE non-preemptible TLS.
  (`D101382 <https://reviews.llvm.org/D101382>`_)
* The PowerPC port now supports ``--power10-stubs=no`` to omit Power10 instructions from call stubs.
  (`D94625 <https://reviews.llvm.org/D94625>`_)
* Fixed a thunk creation bug in the PowerPC port when TOC/NOTOC calls are mixed.
  (`D101837 <https://reviews.llvm.org/D101837>`_)
* The RISC-V port now resolves undefined weak relocations to the current location if not using PLT.
  (`D103001 <https://reviews.llvm.org/D103001>`_)
* ``R_386_GOTOFF`` relocations from .debug_info are now allowed to be compatible with GCC.
  (`D95994 <https://reviews.llvm.org/D95994>`_)
* ``gotEntrySize`` has been added to improve support for the ILP32 ABI of x86-64.
  (`D102569 <https://reviews.llvm.org/D102569>`_)

Breaking changes
----------------

* ``--shuffle-sections=<seed>`` has been changed to ``--shuffle-sections=<section-glob>=<seed>``.
  Specify ``*`` as ``<section-glob>`` to get the previous behavior.

COFF Improvements
-----------------

* Avoid thread exhaustion when running on 32 bit Windows.
  (`D105506 <https://reviews.llvm.org/D105506>`_)

* Improve terminating the process on Windows while a thread pool might be
  running. (`D102944 <https://reviews.llvm.org/D102944>`_)

MinGW Improvements
------------------

* Support for linking directly against a DLL without using an import library
  has been added. (`D104530 <https://reviews.llvm.org/D104530>`_ and
  `D104531 <https://reviews.llvm.org/D104531>`_)

* Fix linking with ``--export-all-symbols`` in combination with
  ``-function-sections``. (`D101522 <https://reviews.llvm.org/D101522>`_ and
  `D101615 <https://reviews.llvm.org/D101615>`_)

* Fix automatic export of symbols from LTO objects.
  (`D101569 <https://reviews.llvm.org/D101569>`_)

* Accept more spellings of some options.
  (`D107237 <https://reviews.llvm.org/D107237>`_ and
  `D107253 <https://reviews.llvm.org/D107253>`_)

Mach-O Improvements
-------------------

The Mach-O backend is now able to link several large, real-world programs,
though we are still working out the kinks.

* arm64 is now supported as a target. (`D88629 <https://reviews.llvm.org/D88629>`_)
* arm64_32 is now supported as a target. (`D99822 <https://reviews.llvm.org/D99822>`_)
* Branch-range-extension thunks are now supported. (`D100818 <https://reviews.llvm.org/D100818>`_)
* ``-dead_strip`` is now supported. (`D103324 <https://reviews.llvm.org/D103324>`_)
* Support for identical code folding (``--icf=all``) has been added.
  (`D103292 <https://reviews.llvm.org/D103292>`_)
* Support for special ``$start`` and ``$end`` symbols for segment & sections has been
  added. (`D106767 <https://reviews.llvm.org/D106767>`_, `D106629 <https://reviews.llvm.org/D106629>`_)
* ``$ld$previous`` symbols are now supported. (`D103505 <https://reviews.llvm.org/D103505 >`_)
* ``$ld$install_name`` symbols are now supported. (`D103746 <https://reviews.llvm.org/D103746>`_)
* ``__mh_*_header`` symbols are now supported. (`D97007 <https://reviews.llvm.org/D97007>`_)
* LC_CODE_SIGNATURE is now supported. (`D96164 <https://reviews.llvm.org/D96164>`_)
* LC_FUNCTION_STARTS is now supported. (`D97260 <https://reviews.llvm.org/D97260>`_)
* LC_DATA_IN_CODE is now supported. (`D103006 <https://reviews.llvm.org/D103006>`_)
* Bind opcodes are more compactly encoded. (`D106128 <https://reviews.llvm.org/D106128>`_,
  `D105075 <https://reviews.llvm.org/D105075>`_)
* LTO cache support has been added. (`D105922 <https://reviews.llvm.org/D105922>`_)
* ``-application_extension`` is now supported. (`D105818 <https://reviews.llvm.org/D105818>`_)
* ``-export_dynamic`` is now partially supported. (`D105482 <https://reviews.llvm.org/D105482>`_)
* ``-arch_multiple`` is now supported. (`D105450 <https://reviews.llvm.org/D105450>`_)
* ``-final_output`` is now supported. (`D105449 <https://reviews.llvm.org/D105449>`_)
* ``-umbrella`` is now supported. (`D105448 <https://reviews.llvm.org/D105448>`_)
* ``--print-dylib-search`` is now supported. (`D103985 <https://reviews.llvm.org/D103985>`_)
* ``-force_load_swift_libs`` is now supported. (`D103709 <https://reviews.llvm.org/D103709>`_)
* ``-reexport_framework``, ``-reexport_library``, ``-reexport-l`` are now supported.
  (`D103497 <https://reviews.llvm.org/D103497>`_)
* ``.weak_def_can_be_hidden`` is now supported. (`D101080 <https://reviews.llvm.org/D101080>`_)
* ``-add_ast_path`` is now supported. (`D100076 <https://reviews.llvm.org/D100076>`_)
* ``-segprot`` is now supported.  (`D99389 <https://reviews.llvm.org/D99389>`_)
* ``-dependency_info`` is now partially supported. (`D98559 <https://reviews.llvm.org/D98559>`_)
* ``--time-trace`` is now supported. (`D98419 <https://reviews.llvm.org/D98419>`_)
* ``-mark_dead_strippable_dylib`` is now supported. (`D98262 <https://reviews.llvm.org/D98262>`_)
* ``-[un]exported_symbol[s_list]`` is now supported. (`D98223 <https://reviews.llvm.org/D98223>`_)
* ``-flat_namespace`` is now supported. (`D97641 <https://reviews.llvm.org/D97641>`_)
* ``-rename_section`` and ``-rename_segment`` are now supported. (`D97600 <https://reviews.llvm.org/D97600>`_)
* ``-bundle_loader`` is now supported. (`D95913 <https://reviews.llvm.org/D95913>`_)
* ``-map`` is now partially supported. (`D98323 <https://reviews.llvm.org/D98323>`_)

There were numerous other bug-fixes as well.

WebAssembly Improvements
------------------------

