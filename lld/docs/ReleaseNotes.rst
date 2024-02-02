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

* ``--fat-lto-objects`` option is added to support LLVM FatLTO.
  Without ``--fat-lto-objects``, LLD will link LLVM FatLTO objects using the
  relocatable object file. (`D146778 <https://reviews.llvm.org/D146778>`_)
* ``-Bsymbolic-non-weak`` is added to directly bind non-weak definitions.
  (`D158322 <https://reviews.llvm.org/D158322>`_)
* ``--lto-validate-all-vtables-have-type-infos``, which complements
  ``--lto-whole-program-visibility``, is added to disable unsafe whole-program
  devirtualization. ``--lto-known-safe-vtables=<glob>`` can be used
  to mark known-safe vtable symbols.
  (`D155659 <https://reviews.llvm.org/D155659>`_)
* ``--no-allow-shlib-undefined`` now reports errors for DSO referencing
  non-exported definitions.
  (`#70769 <https://github.com/llvm/llvm-project/pull/70769>`_)
* common-page-size can now be larger than the system page-size.
  (`#57618 <https://github.com/llvm/llvm-project/issues/57618>`_)
* When call graph profile information is availablue due to instrumentation or
  sample PGO, input sections are now sorted using the new ``cdsort`` algorithm,
  better than the previous ``hfsort`` algorithm.
  (`D152840 <https://reviews.llvm.org/D152840>`_)
* Symbol assignments like ``a = DEFINED(a) ? a : 0;`` are now handled.
  (`#65866 <https://github.com/llvm/llvm-project/pull/65866>`_)
* ``OVERLAY`` now supports optional start address and LMA
  (`#77272 <https://github.com/llvm/llvm-project/pull/77272>`_)
* Relocations referencing a symbol defined in ``/DISCARD/`` section now lead to
  an error.
  (`#69295 <https://github.com/llvm/llvm-project/pull/69295>`_)
* For AArch64 MTE, global variable descriptors have been implemented.
  (`D152921 <https://reviews.llvm.org/D152921>`_)
* ``R_LARCH_PCREL20_S2``/``R_LARCH_ADD6``/``R_LARCH_CALL36`` and extreme code
  model relocations are now supported.
* ``--emit-relocs`` is now supported for RISC-V linker relaxation.
  (`D159082 <https://reviews.llvm.org/D159082>`_)
* Call relaxation respects RVC when mixing +c and -c relocatable files.
  (`#73977 <https://github.com/llvm/llvm-project/pull/73977>`_)
* ``R_RISCV_SET_ULEB128``/``R_RISCV_SUB_ULEB128`` relocations are now supported.
  (`#72610 <https://github.com/llvm/llvm-project/pull/72610>`_)
  (`#77261 <https://github.com/llvm/llvm-project/pull/77261>`_)
* RISC-V TLSDESC is now supported.
  (`#79239 <https://github.com/llvm/llvm-project/pull/79239>`_)

Breaking changes
----------------

COFF Improvements
-----------------

* Added support for ``--time-trace`` and associated ``--time-trace-granularity``.
  This generates a .json profile trace of the linker execution.

MinGW Improvements
------------------

MachO Improvements
------------------

WebAssembly Improvements
------------------------

* Indexes are no longer required on archive files.  Instead symbol information
  is read from object files within the archive.  This matches the behaviour of
  the ELF linker.

Fixes
#####
