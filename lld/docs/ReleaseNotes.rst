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
* ``--package-metadata=`` has been added to create package metadata notes
  (`D131439 <https://reviews.llvm.org/D131439>`_)

Breaking changes
----------------

COFF Improvements
-----------------

* ...

MinGW Improvements
------------------

* The ``--exclude-symbols`` option is now supported.
  (`D130118 <https://reviews.llvm.org/D130118>`_)

* Support for an entirely new object file directive, ``-exclude-symbols:``,
  has been implemented. (`D130120 <https://reviews.llvm.org/D130120>`_)

MachO Improvements
------------------

WebAssembly Improvements
------------------------

