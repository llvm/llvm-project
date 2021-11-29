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
* If ``-Map`` is specified, ``--cref`` will be printted to the specified file.
  (`D114663 <https://reviews.llvm.org/D114663>`_)

Architecture specific changes:

* The x86-32 port now supports TLSDESC (``-mtls-dialect=gnu2``).
  (`D112582 <https://reviews.llvm.org/D112582>`_)
* The x86-64 port now handles non-RAX/non-adjacent ``R_X86_64_GOTPC32_TLSDESC``
  and ``R_X86_64_TLSDESC_CALL`` (``-mtls-dialect=gnu2``).
  (`D114416 <https://reviews.llvm.org/D114416>`_)
* For x86-64, ``--no-relax`` now suppresses ``R_X86_64_GOTPCRELX`` and
  ``R_X86_64_REX_GOTPCRELX`` GOT optimization
  (`D113615 <https://reviews.llvm.org/D113615>`_)

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

