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

* ``ELFCOMPRESS_ZSTD`` compressed input sections are now supported.
  (`D129406 <https://reviews.llvm.org/D129406>`_)
* ``--compress-debug-sections=zstd`` is now available to compress debug
  sections with zstd (``ELFCOMPRESS_ZSTD``).
  (`D133548 <https://reviews.llvm.org/D133548>`_)

Breaking changes
----------------

COFF Improvements
-----------------

* ...

MinGW Improvements
------------------

* The lld-specific options ``--guard-cf``, ``--no-guard-cf``,
  ``--guard-longjmp`` and ``--no-guard-longjmp`` has been added to allow
  enabling Control Flow Guard and long jump hardening. These options are
  disabled by default, but enabling ``--guard-cf`` will also enable
  ``--guard-longjmp`` unless ``--no-guard-longjmp`` is also specified.
  ``--guard-longjmp`` depends on ``--guard-cf`` and cannot be used by itself.
  Note that these features require the ``_load_config_used`` symbol to contain
  the load config directory and be filled with the required symbols.
  (`D132808 <https://reviews.llvm.org/D132808>`_)

MachO Improvements
------------------

WebAssembly Improvements
------------------------

