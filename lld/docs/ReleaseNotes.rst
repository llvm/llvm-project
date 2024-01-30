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

* ``--fat-lto-objects`` option is added to support LLVM FatLTO.
  Without ``--fat-lto-objects``, LLD will link LLVM FatLTO objects using the
  relocatable object file. (`D146778 <https://reviews.llvm.org/D146778>`_)
* common-page-size can now be larger than the system page-size.
  (`#57618 <https://github.com/llvm/llvm-project/issues/57618>`_)

Breaking changes
----------------

COFF Improvements
-----------------

* Added support for ``--time-trace`` and associated ``--time-trace-granularity``.
  This generates a .json profile trace of the linker execution.

* Prefer library paths specified with ``-libpath:`` over the implicitly
  detected toolchain paths.

MinGW Improvements
------------------

* Added support for many LTO and ThinLTO options.

* LLD no longer tries to autodetect and pick up MSVC/WinSDK installations
  when run in MinGW mode.

* The ``--icf=safe`` option now works as expected; it was previously a no-op.

* More correctly handle LTO of files that define ``__imp_`` prefixed dllimport
  redirections.

* The strip flags ``-S`` and ``-s`` now can be used to strip out DWARF debug
  info and symbol tables while emitting a PDB debug info file.

MachO Improvements
------------------

WebAssembly Improvements
------------------------

* Indexes are no longer required on archive files.  Instead symbol information
  is read from object files within the archive.  This matches the behaviour of
  the ELF linker.

Fixes
#####
