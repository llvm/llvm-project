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

Breaking changes
----------------

COFF Improvements
-----------------

MinGW Improvements
------------------

MachO Improvements
------------------

WebAssembly Improvements
------------------------

Fixes
#####

* Arm exception index tables (.ARM.exidx sections) are now ouptut
  correctly when they are at a non zero offset within their output
  section. (`D148033 <https://reviews.llvm.org/D148033>`_)
