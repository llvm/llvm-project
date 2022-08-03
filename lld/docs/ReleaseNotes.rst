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

* ``-z pack-relative-relocs`` is now available to support ``DT_RELR`` for glibc 2.36+.
  (`D120701 <https://reviews.llvm.org/D120701>`_)
* ``--no-fortran-common`` (pre 12.0.0 behavior) is now the default.
* ``--load-pass-plugin`` has been added to load a new pass manager plugin.
  (`D120490 <https://reviews.llvm.org/D120490>`_)
* ``--android-memtag-{mode=,stack,heap}`` have been added to synthesize SHT_NOTE for memory tags on Android.
  (`D119384 <https://reviews.llvm.org/D119384>`_)
* ``FORCE_LLD_DIAGNOSTICS_CRASH`` environment variable is now available to force LLD to crash.
  (`D128195 <https://reviews.llvm.org/D128195>`_)
* ``--wrap`` semantics have been refined.
  (`rG7288b85cc80f1ce5509aeea860e6b4232cd3ca01 <https://reviews.llvm.org/rG7288b85cc80f1ce5509aeea860e6b4232cd3ca01>`_)
  (`D118756 <https://reviews.llvm.org/D118756>`_)
  (`D124056 <https://reviews.llvm.org/D124056>`_)
* ``--build-id={md5,sha1}`` are now implemented with truncated BLAKE3.
  (`D121531 <https://reviews.llvm.org/D121531>`_)
* ``--emit-relocs``: ``.rel[a].eh_frame`` relocation offsets are now adjusted.
  (`D122459 <https://reviews.llvm.org/D122459>`_)
* ``--emit-relocs``: fixed missing ``STT_SECTION`` when the first input section is synthetic.
  (`D122463 <https://reviews.llvm.org/D122463>`_)
* ``(TYPE=<value>)`` can now be used in linker scripts.
  (`D118840 <https://reviews.llvm.org/D118840>`_)
* Local symbol initialization is now performed in parallel.
  (`D119909 <https://reviews.llvm.org/D119909>`_)
  (`D120626 <https://reviews.llvm.org/D120626>`_)

Breaking changes
----------------

* Archives are now parsed as ``--start-lib`` object files. If a member is neither
  an ELF relocatable object file nor an LLVM bitcode file, ld.lld will give a warning.
  (`D119074 <https://reviews.llvm.org/D119074>`_)
* The GNU ld incompatible ``--no-define-common`` has been removed.
* The obscure ``-dc``/``-dp`` options have been removed.
  (`D119108 <https://reviews.llvm.org/D119108>`_)
* ``-d`` is now ignored.
* If a prevailing COMDAT group defines STB_WEAK symbol, having a STB_GLOBAL symbol in a non-prevailing group is now rejected with a diagnostic.
  (`D120626 <https://reviews.llvm.org/D120626>`_)
* Support for the legacy ``.zdebug`` format has been removed. Run
  ``objcopy --decompress-debug-sections`` in case old object files use ``.zdebug``.
  (`D126793 <https://reviews.llvm.org/D126793>`_)
* ``--time-trace-file=<file>`` has been removed.
  Use ``--time-trace=<file>`` instead.
  (`D128451 <https://reviews.llvm.org/D128451>`_)

COFF Improvements
-----------------

* Added autodetection of MSVC toolchain, a la clang-cl.  Also added
  ``/winsysroot:`` support for explicit specification of MSVC toolchain
  location, similar to clang-cl's ``/winsysroot``. For now,
  ``/winsysroot:`` requires also passing in an explicit ``/machine:`` flag.
  (`D118070 <https://reviews.llvm.org/D118070>`_)
* ...

MinGW Improvements
------------------

* The ``--disable-reloc-section`` option is now supported.
  (`D127478 <https://reviews.llvm.org/D127478>`_)

MachO Improvements
------------------

* We now support proper relocation and pruning of EH frames. **Note:** this
  comes at some performance overhead on x86_64 builds, and we recommend adding
  the ``-femit-compact-unwind=no-compact-unwind`` compile flag to avoid it.
  (`D129540 <https://reviews.llvm.org/D129540>`_,
  `D122258 <https://reviews.llvm.org/D122258>`_)

New flags
#########

* ``-load_hidden`` and ``-hidden-l`` are now supported.
  (`D130473 <https://reviews.llvm.org/D130473>`_,
  `D130529 <https://reviews.llvm.org/D130529>`_)
* ``-alias`` is now supported. (`D129938 <https://reviews.llvm.org/D129938>`_)
* ``-no_exported_symbols`` and  ``-exported_symbols_list <empty file>`` are now
  supported. (`D127562 <https://reviews.llvm.org/D127562>`_)
* ``-w`` -- to suppress warnings -- is now supported.
  (`D127564 <https://reviews.llvm.org/D127564>`_)
* ``-non_global_symbols_strip_list``, ``-non_global_symbols_no_strip_list``, and
  ``-x`` are now supported. (`D126046 <https://reviews.llvm.org/D126046>`_)
* ``--icf=safe`` is now supported.
  (`D128938 <https://reviews.llvm.org/D128938>`_,
  `D123752 <https://reviews.llvm.org/D123752>`_)
* ``-why_live`` is now supported.
  (`D120377 <https://reviews.llvm.org/D120377>`_)
* ``-pagezero_size`` is now supported.
  (`D118724 <https://reviews.llvm.org/D118724>`_)

Improvements
############

* Linker optimization hints are now supported.
  (`D129427 <https://reviews.llvm.org/D129427>`_,
  `D129059 <https://reviews.llvm.org/D129059>`_,
  `D128942 <https://reviews.llvm.org/D128942>`_,
  `D128093 <https://reviews.llvm.org/D128093>`_)
* Rebase opcodes are now encoded more compactly.
  (`D130180 <https://reviews.llvm.org/D130180>`_,
  `D128798 <https://reviews.llvm.org/D128798>`_)
* C-strings are now aligned more compactly.
  (`D121342 <https://reviews.llvm.org/D121342>`_)
* ``--deduplicate-literals`` (and ``--icf={safe,all}``) now fold the
  ``__cfstring`` section.
  (`D130134  <https://reviews.llvm.org/D130134>`_,
  `D120137 <https://reviews.llvm.org/D120137>`_)
* ICF now folds the ``__objc_classrefs`` section.
  (`D121053 <https://reviews.llvm.org/D121053>`_)
* ICF now folds functions with identical LSDAs.
  (`D129830 <https://reviews.llvm.org/D129830>`_)
* STABS entries for folded functions are now omitted.
  (`D123252 <https://reviews.llvm.org/D123252>`_)
* ``__objc_imageinfo`` sections are now folded.
  (`D130125 <https://reviews.llvm.org/D130125>`_)
* Dylibs with ``LC_DYLD_EXPORTS_TRIE`` can now be read.
  (`D129430 <https://reviews.llvm.org/D129430>`_)
* Writing zippered dylibs is now supported.
  (`D124887 <https://reviews.llvm.org/D124887>`_)
* C-string literals are now included in the mapfile.
  (`D118077 <https://reviews.llvm.org/D118077>`_)
* Symbol names in several more diagnostics are now demangled.
  (`D130490 <https://reviews.llvm.org/D130490>`_,
  `D127110 <https://reviews.llvm.org/D127110>`_,
  `D125732 <https://reviews.llvm.org/D125732>`_)
* Source information is now included in symbol error messages.
  (`D128425 <https://reviews.llvm.org/D128425>`_,
  `D128184 <https://reviews.llvm.org/D128184>`_)
* Numerous other improvements were made to diagnostic messages.
  (`D127753 <https://reviews.llvm.org/D127753>`_,
  `D127696 <https://reviews.llvm.org/D127696>`_,
  `D127670 <https://reviews.llvm.org/D127670>`_,
  `D118903 <https://reviews.llvm.org/D118903>`_,
  `D118798 <https://reviews.llvm.org/D118798>`_)
* Many performance and memory improvements were made.
  (`D130000 <https://reviews.llvm.org/D130000>`_,
  `D128298 <https://reviews.llvm.org/D128298>`_,
  `D128290 <https://reviews.llvm.org/D128290>`_,
  `D126800 <https://reviews.llvm.org/D126800>`_,
  `D126785 <https://reviews.llvm.org/D126785>`_,
  `D121052 <https://reviews.llvm.org/D121052>`_)
* Order files and call graph sorting can now be used together.
  (`D117354 <https://reviews.llvm.org/D117354>`_)
* Give LTO more precise symbol resolutions, which allows optimizations to be
  more effective.
  (`D119506 <https://reviews.llvm.org/D119506>`_,
  `D119372 <https://reviews.llvm.org/D119372>`_,
  `D119767 <https://reviews.llvm.org/D119767>`_)
* Added partial support for linking object files built with DTrace probes.
  (`D129062 <https://reviews.llvm.org/D129062>`_)

Fixes
#####

* Programs using Swift linked with the 14.0 SDK but an older deployment target
  no longer crash at startup when running on older iOS versions. This is because
  we now correctly support ``$ld$previous`` symbols that contain an explicit
  symbol name. (`D130725 <https://reviews.llvm.org/D130725>`_)
* Match ld64's behavior when an archive is specified both via
  ``LC_LINKER_OPTION`` and via the command line.
  (`D129556 <https://reviews.llvm.org/D129556>`_)
* ``-ObjC`` now correctly loads archives with Swift sections.
  (`D125250 <https://reviews.llvm.org/D125250>`_)
* ``-lto_object_path`` now accepts a filename (instead of just a directory
  name.) (`D129705 <https://reviews.llvm.org/D129705>`_)
* The ``LC_UUID`` hash now includes the output file's name.
  (`D122843 <https://reviews.llvm.org/D122843>`_)
* ``-flat_namespace`` now correctly makes all extern symbols in a dylib
  interposable. (`D119294 <https://reviews.llvm.org/D119294>`_)
* Fixed compact unwind output when linking on 32-bit hosts.
  (`D129363 <https://reviews.llvm.org/D129363>`_)
* Exporting private symbols no longer triggers an assertion.
  (`D124143 <https://reviews.llvm.org/D124143>`_)
* MacOS-only ``.tbd`` files are now supported when targeting Catalyst.
  (`D124336 <https://reviews.llvm.org/D124336>`_)
* Thunk symbols now have local visibility, avoiding false duplicate symbol
  errors. (`D122624 <https://reviews.llvm.org/D122624>`_)
* Fixed handling of relocatable object files within frameworks.
  (`D114841 <https://reviews.llvm.org/D114841>`_)

WebAssembly Improvements
------------------------

