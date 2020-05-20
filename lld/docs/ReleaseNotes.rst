========================
lld 11.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 11.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release 11.0.0.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* New ``--time-trace`` option records a time trace file that can be viewed in
  chrome://tracing. The file can be specified with ``--time-trace-file``.
  Trace granularity can be specified with ``--time-trace-granularity``.
  (`D71060 <https://reviews.llvm.org/D71060>`_)
* ...

Breaking changes
----------------

* One-dash form of some long option (``--thinlto-*``, ``--lto-*``, ``--shuffle-sections=``)
  are no longer supported.

COFF Improvements
-----------------

* ...

MinGW Improvements
------------------

* ...

MachO Improvements
------------------

* Item 1.

WebAssembly Improvements
------------------------

