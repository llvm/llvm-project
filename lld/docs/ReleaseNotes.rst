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

* ``--compress-sections <section-glib>={none,zlib,zstd}[:level]`` is added to compress
  matched output sections without the ``SHF_ALLOC`` flag.
  (`#84855 <https://github.com/llvm/llvm-project/pull/84855>`_)
  (`#90567 <https://github.com/llvm/llvm-project/pull/90567>`_)
* The default compression level for zlib is now independent of linker
  optimization level (``Z_BEST_SPEED``).
* ``GNU_PROPERTY_AARCH64_FEATURE_PAUTH`` notes, ``R_AARCH64_AUTH_ABS64`` and
  ``R_AARCH64_AUTH_RELATIVE`` relocations are now supported.
  (`#72714 <https://github.com/llvm/llvm-project/pull/72714>`_)
* ``--debug-names`` is added to create a merged ``.debug_names`` index
  from input ``.debug_names`` sections. Type units are not handled yet.
  (`#86508 <https://github.com/llvm/llvm-project/pull/86508>`_)
* ``--enable-non-contiguous-regions`` option allows automatically packing input
  sections into memory regions by automatically spilling to later matches if a
  region would overflow. This reduces the toil of manually packing regions
  (typical for embedded). It also makes full LTO feasible in such cases, since
  IR merging currently prevents the linker script from referring to input
  files. (`#90007 <https://github.com/llvm/llvm-project/pull/90007>`_)
* ``--force-group-allocation`` is implemented to discard ``SHT_GROUP`` sections
  and combine relocation sections if their relocated section group members are
  placed to the same output section.
  (`#94704 <https://github.com/llvm/llvm-project/pull/94704>`_)
* ``--build-id`` now defaults to generating a 20-byte digest ("sha1") instead
  of 8-byte ("fast"). This improves compatibility with RPM packaging tools.
  (`#93943 <https://github.com/llvm/llvm-project/pull/93943>`_)

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
