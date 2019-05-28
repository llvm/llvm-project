=======================
lld 9.0.0 Release Notes
=======================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 9.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release 9.0.0.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* ld.lld now has typo suggestions for flags:
  ``$ ld.lld --call-shared`` now prints
  ``unknown argument '--call-shared', did you mean '--call_shared'``.

* ...

COFF Improvements
-----------------

* Like the ELF driver, lld-link now has typo suggestions for flags.

* lld-link now correctly reports duplicate symbol errors for obj files
  that were compiled with /Gy.

* lld-link now correctly reports duplicate symbol errors when several res
  input files define resources with the same type, name, and language.
  This can be demoted to a warning using ``/force:multipleres``.

* Having more than two ``/natvis:`` now works correctly; it used to not
  work for larger binaries before.

* Undefined symbols are now printed only in demangled form. Pass
  ``/demangle:no`` to see raw symbol names instead.

* The following flags have been added: ``/functionpadmin``, ``/swaprun:``,
  ``/threads:no``

* Several speed and memory usage improvements.

* ...

MinGW Improvements
------------------

* lld now correctly links crtend.o as the last object file, handling
  terminators for the sections such as .eh_frame properly, fixing
  DWARF exception handling with libgcc and gcc's crtend.o.

MachO Improvements
------------------

* Item 1.

WebAssembly Improvements
------------------------

* ...
