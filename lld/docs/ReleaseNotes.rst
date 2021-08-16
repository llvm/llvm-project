========================
lld 13.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 13.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release 13.0.0.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* ``-Bsymbolic -Bsymbolic-functions`` has been changed to behave the same as ``-Bsymbolic-functions``. This matches GNU ld.
  (`D102461 <https://reviews.llvm.org/D102461>`_)
* ``-Bno-symbolic`` has been added.
  (`D102461 <https://reviews.llvm.org/D102461>`_)
* A new linker script command ``OVERWRITE_SECTIONS`` has been added.
  (`D103303 <https://reviews.llvm.org/D103303>`_)
* ``-Bsymbolic-non-weak-functions`` has been added as a ``STB_GLOBAL`` subset of ``-Bsymbolic-functions``.
  (`D102570 <https://reviews.llvm.org/D102570>`_)

Breaking changes
----------------

* ``--shuffle-sections=<seed>`` has been changed to ``--shuffle-sections=<section-glob>=<seed>``.
  Specify ``*`` as ``<section-glob>`` to get the previous behavior.

COFF Improvements
-----------------

* Avoid thread exhaustion when running on 32 bit Windows.
  (`D105506 <https://reviews.llvm.org/D105506>`_)

* Improve terminating the process on Windows while a thread pool might be
  running. (`D102944 <https://reviews.llvm.org/D102944>`_)

MinGW Improvements
------------------

* Support for linking directly against a DLL without using an import library
  has been added. (`D104530 <https://reviews.llvm.org/D104530>`_ and
  `D104531 <https://reviews.llvm.org/D104531>`_)

* Fix linking with ``--export-all-symbols`` in combination with
  ``-function-sections``. (`D101522 <https://reviews.llvm.org/D101522>`_ and
  `D101615 <https://reviews.llvm.org/D101615>`_)

* Fix automatic export of symbols from LTO objects.
  (`D101569 <https://reviews.llvm.org/D101569>`_)

* Accept more spellings of some options.
  (`D107237 <https://reviews.llvm.org/D107237>`_ and
  `D107253 <https://reviews.llvm.org/D107253>`_)

MachO Improvements
------------------

* Item 1.

WebAssembly Improvements
------------------------

