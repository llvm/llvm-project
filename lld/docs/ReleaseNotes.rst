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
* ``FORCE_LLD_DIAGNOSTICS_CRASH`` environment variable is now available to force LLD to crash.
  (`D128195 <https://reviews.llvm.org/D128195>`_)

Breaking changes
----------------

* The GNU ld incompatible ``--no-define-common`` has been removed.
* The obscure ``-dc``/``-dp`` options have been removed.
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

* ...

MachO Improvements
------------------

* We now support proper relocation and pruning of EH frames. **Note:** this
  comes at some performance overhead on x86_64 builds, and we recommend adding
  the ``-femit-compact-unwind=no-compact-unwind`` compile flag to avoid it.
  (`D129540 <https://reviews.llvm.org/D129540>`_,
  `D122258 <https://reviews.llvm.org/D122258>`_)

WebAssembly Improvements
------------------------

