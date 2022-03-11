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

Breaking changes
----------------

* The GNU ld incompatible ``--no-define-common`` has been removed.
* The obscure ``-dc``/``-dp`` options have been removed.
* ``-d`` is now ignored.

COFF Improvements
-----------------

* Added autodetection of MSVC toolchain, a la clang-cl.  Also added /winsysroot
  support for explicit specification of MSVC toolchain location.
  (`D118070 <https://reviews.llvm.org/D118070>`_)
* ...

MinGW Improvements
------------------

* ...

MachO Improvements
------------------

* Item 1.

WebAssembly Improvements
------------------------

