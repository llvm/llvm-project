========================
lld 12.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 12.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release 12.0.0.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* ``--error-handling-script`` is added to allow for user-defined handlers upon
  missing libraries. (`D87758 <https://reviews.llvm.org/D87758>`_)

Breaking changes
----------------

* ...

COFF Improvements
-----------------

* Error out clearly if creating a DLL with too many exported symbols.
  (`D86701 <https://reviews.llvm.org/D86701>`_)

MinGW Improvements
------------------

* Enabled dynamicbase by default. (`D86654 <https://reviews.llvm.org/D86654>`_)

* Tolerate mismatches between COMDAT section sizes with different amount of
  padding (produced by binutils) by inspecting the aux section definition.
  (`D86659 <https://reviews.llvm.org/D86659>`_)

* Support setting the subsystem version via the subsystem argument.
  (`D88804 <https://reviews.llvm.org/D88804>`_)

* Implemented the GNU -wrap option.
  (`D89004 <https://reviews.llvm.org/D89004>`_,
  `D91689 <https://reviews.llvm.org/D91689>`_)

* Handle the ``--demangle`` and ``--no-demangle`` options.
  (`D93950 <https://reviews.llvm.org/D93950>`_)


MachO Improvements
------------------

* Item 1.

WebAssembly Improvements
------------------------

