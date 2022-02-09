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

* ``--export-dynamic-symbol-list`` has been added.
  (`D107317 <https://reviews.llvm.org/D107317>`_)
* ``--why-extract`` has been added to query why archive members/lazy object files are extracted.
  (`D109572 <https://reviews.llvm.org/D109572>`_)
* If ``-Map`` is specified, ``--cref`` will be printed to the specified file.
  (`D114663 <https://reviews.llvm.org/D114663>`_)
* ``-z bti-report`` and ``-z cet-report`` are now supported.
  (`D113901 <https://reviews.llvm.org/D113901>`_)
* ``--lto-pgo-warn-mismatch`` has been added.
  (`D104431 <https://reviews.llvm.org/D104431>`_)
* Archives without an index (symbol table) are now supported and work with
  ``--warn-backrefs``. One may build such an archive with ``llvm-ar rcS
  [--thin]`` to save space.
  (`D117284 <https://reviews.llvm.org/D117284>`_)
  The archive index may be `entirely ignored <https://reviews.llvm.org/D119074>`
  in a future release.
* No longer deduplicate local symbol names at the default optimization level of ``-O1``.
  This results in a larger ``.strtab`` (usually less than 1%) but a faster link
  time. Use optimization level ``-O2`` to restore the deduplication. The ``-O2``
  deduplication may be dropped in the future to help parallel ``.symtab`` write.
* In relocatable output, relocations to discarded symbols now use tombstone
  values.
  (`D116946 <https://reviews.llvm.org/D116946>`_)
* Orphan section placement now picks a more suitable segment. Previously the
  algorithm might pick a readonly segment for a writable orphan section and make
  the segment writable.
  (`D111717 <https://reviews.llvm.org/D111717>`_)
* An empty output section moved by an ``INSERT`` comment now gets appropriate
  flags.
  (`D118529 <https://reviews.llvm.org/D118529>`_)
* Negation in a memory region attribute is now correctly handled.
  (`D113771 <https://reviews.llvm.org/D113771>`_)
* ``--compress-debug-sections=zlib`` is now run in parallel. ``{clang,gcc} -gz`` link
  actions are significantly faster.
  (`D117853 <https://reviews.llvm.org/D117853>`_)
* "relocation out of range" diagnostics and a few uncommon diagnostics
  now report an object file location beside a source file location.
  (`D112518 <https://reviews.llvm.org/D117853>`_)
* The write of ``.rela.dyn`` and ``SHF_MERGE|SHF_STRINGS`` sections (e.g.
  ``.debug_str``) is now run in parallel.

Architecture specific changes:

* The AArch64 port now supports adrp+ldr and adrp+add optimizations.
  ``--no-relax`` can suppress the optimization.
  (`D112063 <https://reviews.llvm.org/D112063>`_)
  (`D117614 <https://reviews.llvm.org/D117614>`_)
* The x86-32 port now supports TLSDESC (``-mtls-dialect=gnu2``).
  (`D112582 <https://reviews.llvm.org/D112582>`_)
* The x86-64 port now handles non-RAX/non-adjacent ``R_X86_64_GOTPC32_TLSDESC``
  and ``R_X86_64_TLSDESC_CALL`` (``-mtls-dialect=gnu2``).
  (`D114416 <https://reviews.llvm.org/D114416>`_)
* The x86-32 and x86-64 ports now support mixed TLSDESC and TLS GD, i.e. mixing
  objects compiled with and without ``-mtls-dialect=gnu2`` referencing the same
  TLS variable is now supported.
  (`D114416 <https://reviews.llvm.org/D114416>`_)
* For x86-64, ``--no-relax`` now suppresses ``R_X86_64_GOTPCRELX`` and
  ``R_X86_64_REX_GOTPCRELX`` GOT optimization
  (`D113615 <https://reviews.llvm.org/D113615>`_)
* ``R_X86_64_PLTOFF64`` is now supported.
  (`D112386 <https://reviews.llvm.org/D112386>`_)
* ``R_AARCH64_NONE``, ``R_PPC_NONE``, and ``R_PPC64_NONE`` in input REL
  relocation sections are now supported.

Breaking changes
----------------

* ``e_entry`` no longer falls back to the address of ``.text`` if the entry symbol does not exist.
  Instead, a value of 0 will be written.
  (`D110014 <https://reviews.llvm.org/D110014>`_)
* ``--lto-pseudo-probe-for-profiling`` has been removed. In LTO, the compiler
  enables this feature automatically.
  (`D110209 <https://reviews.llvm.org/D110209>`_)
* Use of ``--[no-]define-common``, ``-d``, ``-dc``, and ``-dp`` will now get a
  warning. They will be removed or ignored in 15.0.0.
  (`llvm-project#53660 <https://github.com/llvm/llvm-project/issues/53660>`_)

COFF Improvements
-----------------

* Correctly handle a signed immediate offset in ARM64 adr/adrp relocations.
  (`D114347 <https://reviews.llvm.org/D114347>`_)

* Omit section and label symbols from the symbol table.
  (`D113866 <https://reviews.llvm.org/D113866>`_)

MinGW Improvements
------------------

* ``--heap`` is now handled.
  (`D118405 <https://reviews.llvm.org/D118405>`_)

MachO Improvements
------------------

* Item 1.

WebAssembly Improvements
------------------------

