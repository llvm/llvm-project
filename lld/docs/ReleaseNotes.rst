========================
lld 14.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 14.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release 14.0.0.
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
* ``e_entry`` no longer falls back to the address of ``.text`` if the entry symbol does not exist.
  Instead, a value of 0 will be written.
  (`D110014 <https://reviews.llvm.org/D110014>`_)

Breaking changes
----------------

* ...

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

