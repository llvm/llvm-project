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

* Input file loading is now parallelized, meaningfully reducing link time
  for large links.
  (`#191690 <https://github.com/llvm/llvm-project/pull/191690>`_)
* ``--gc-sections`` mark phase is now parallelized.
  (`#189321 <https://github.com/llvm/llvm-project/pull/189321>`_)
* Relocation scanning was rewritten as target-specific scanners for all
  targets with shared library support, devirtualizing the hot
  relocation-classification path.

* Added ``--bp-compression-sort-section=<glob>[=<layout_priority>[=<match_priority>]]``,
  replacing the old coarse ``--bp-compression-sort`` modes with a way to split
  input sections into multiple compression groups, run balanced partitioning
  independently per group, and leave out sections that are poor candidates for
  BP.
  ``layout_priority`` controls group placement order (lower value = placed
  first, default 0). ``match_priority`` resolves conflicts when multiple globs
  match the same section (lower value = higher priority; explicit priority
  beats positional last-match-wins; default: positional). In ELF, the glob
  matches input section names (e.g. ``.text.unlikely.code1``).
* Added ``-z memtag-{mode,heap,stack}`` as generic replacements for the
  Android-specific ``--android-memtag-*`` flags; ``--android-memtag-note``
  keeps the Android-specific memtag note opt-in.
  (`#188205 <https://github.com/llvm/llvm-project/pull/188205>`_)
* Unused space in executable output sections is now filled with trap, primarily
  for ``-z separate-code`` mode.
  (`#176845 <https://github.com/llvm/llvm-project/pull/176845>`_)
* ``.eh_frame_hdr`` now supports the ``DW_EH_PE_sdata8`` encoding,
  auto-upgrading from ``sdata4`` when a table entry or the frame pointer
  exceeds the 32-bit range, instead of erroring out for large executables.
  (`#179089 <https://github.com/llvm/llvm-project/pull/179089>`_)
* ``vna_flags`` is now set to ``VER_FLG_WEAK`` when all undefined references
  to a version are weak, allowing glibc's dynamic loader to warn instead of
  error when the version is missing at runtime.
  (`#176673 <https://github.com/llvm/llvm-project/pull/176673>`_)
* ``.ltext.*`` input sections are now merged into a single ``.ltext`` output
  section, matching the existing ``.ldata.*``/``.lrodata.*``/``.lbss.*``
  handling for the large code model with ``-ffunction-sections``.
  (`#190305 <https://github.com/llvm/llvm-project/pull/190305>`_)
* ``.tbss`` output sections may now use an explicit address expression;
  previously it was silently overridden to follow the preceding ``.tbss``.
  (`#196447 <https://github.com/llvm/llvm-project/pull/196447>`_)
* ``--discard-locals``/``--discard-all`` combined with ``-r``/``--emit-relocs``
  no longer discard local symbols that are referenced only from retained
  non-``SHF_ALLOC`` sections (e.g. ``.L`` symbols referenced by
  ``.debug_info``), fixing DWARF corruption in the output.
  (`#209035 <https://github.com/llvm/llvm-project/pull/209035>`_)
  (`#209042 <https://github.com/llvm/llvm-project/pull/209042>`_)
* ``INCLUDE`` in linker scripts now fully parses its own content instead of
  sharing a lexer buffer stack with the includer, fixing spurious acceptance
  of malformed scripts.
  (`#193427 <https://github.com/llvm/llvm-project/pull/193427>`_)
* The ``OVERLAY`` linker script command now accepts any
  output-section-command (e.g. symbol assignments), not just input section
  descriptions.
  (`#203524 <https://github.com/llvm/llvm-project/pull/203524>`_)
* Thunks are no longer reused across an ``OVERLAY`` boundary unless the
  target output section is guaranteed to be resident at the same time.
  (`#200415 <https://github.com/llvm/llvm-project/pull/200415>`_)
* When a ``SECTIONS`` command interleaves relro and non-relro sections, lld now
  emits one ``PT_GNU_RELRO`` segment per contiguous run of relro sections
  instead of reporting a ``not contiguous with other relro sections`` error.
* ``SHT_NOBITS`` sections are now excluded from LMA overlap checks, matching
  GNU ld and allowing e.g. a startup section to share an LMA with ``.bss``
  in embedded linker scripts.
  (`#196423 <https://github.com/llvm/llvm-project/pull/196423>`_)
* LTO: the middle-end no longer emits new references to, or internalizes,
  symbols defined in bitcode after the extracted-bitcode set has been fixed,
  preventing undefined symbol references from transforms that run after
  linking has determined which bitcode files to extract.
  (`#164916 <https://github.com/llvm/llvm-project/pull/164916>`_)
* DTLTO: significantly improved the performance of adding backend output
  files to the link, especially on Windows.
  (`#186366 <https://github.com/llvm/llvm-project/pull/186366>`_)
* For AArch64, fixed ``.relr.auth.dyn`` -> ``.rela.dyn`` movement to
  properly adjust ``__rela_iplt_start``/``__rela_iplt_end`` and size the
  ``.dynamic`` section for both tags.
  (`#195649 <https://github.com/llvm/llvm-project/pull/195649>`_)
* For AArch64, handle Memtag globals for ``R_AARCH64_AUTH_ABS64``.
  (`#173291 <https://github.com/llvm/llvm-project/pull/173291>`_)
* For AArch64, fixed TLS GD against non-preemptible dynamic symbols (e.g.
  ``protected`` or ``-Bsymbolic``) in DSOs, which previously produced an
  inconsistent GOT entry and spurious preemption.
  (`#207881 <https://github.com/llvm/llvm-project/pull/207881>`_)
* For AArch64, ``adrp``+``ldr`` GOT relaxation is now decided per-symbol,
  all-or-nothing, avoiding invalid relaxation when a branch target sits
  between the ``adrp`` and ``ldr`` of a pair.
  (`#208396 <https://github.com/llvm/llvm-project/pull/208396>`_)
* For AArch64, a redundant local-exec TLS ``add`` with a zero high-12-bits
  immediate is now relaxed to a ``nop``.
  (`#204286 <https://github.com/llvm/llvm-project/pull/204286>`_)
* x86-64 CFI jump table relaxation reduces the runtime overhead of indirect
  calls under Control Flow Integrity by opportunistically moving eligible
  function bodies into the jump table itself.
  (`#147424 <https://github.com/llvm/llvm-project/pull/147424>`_)

Breaking changes
----------------

* The symbol partition feature has been removed. lld no longer recognizes
  ``SHT_LLVM_SYMPART`` sections, which are now treated as ordinary sections. The
  feature saw no adoption beyond a Chromium experiment that has since been
  retired.

* An OutputSection that has an address expression, and is also assigned
  to a MEMORY region, will now use the address expression in preference
  to the next available location in the MEMORY region. This brings LLD
  in line with GNU ld, but is a change in behavior from previous LLD
  releases.
  
* The default extension for time trace files is now ``.time-trace.json``.

COFF Improvements
-----------------

MinGW Improvements
------------------

* Added ``--push-state`` and ``--pop-state``, offering the same semantics as
  when used with the ELF linker: The state of ``--Bstatic``/``--Bdynamic`` and
  ``--whole-archive`` are pushed onto a stack and popped from it.

MachO Improvements
------------------

* ``--bp-compression-sort-section`` now accepts optional layout and match
  priorities (same syntax as ELF). In Mach-O, the glob matches the
  concatenated segment+section name (e.g. ``__TEXT__text``).
* Restructure thunk generation algorithm to be more efficiently create thunks
  (`#193367 <https://github.com/llvm/llvm-project/pull/193367>`_)
* Alphabetically sort LC_LINKER_OPTIONS before processing to match Apple linker behavior
  (`#201604 https://github.com/llvm/llvm-project/pull/201604`)

WebAssembly Improvements
------------------------

Fixes
#####
