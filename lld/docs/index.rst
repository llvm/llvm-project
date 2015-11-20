.. _index:

lld - The LLVM Linker
=====================

lld contains two linkers whose architectures are different from each other.
One is a linker that implements native features directly.
They are in `COFF` or `ELF` directories. Other directories contains the other
implementation that is designed to be a set of modular code for creating
linker tools. This document covers mainly the latter.
For the former, please read README.md in `COFF` directory.

* End-User Features:

  * Compatible with existing linker options
  * Reads standard Object Files (e.g. ELF, Mach-O, PE/COFF)
  * Writes standard Executable Files (e.g. ELF, Mach-O, PE)
  * Remove clang's reliance on "the system linker"
  * Uses the LLVM `"UIUC" BSD-Style license`__.

* Applications:

  * Modular design
  * Support cross linking
  * Easy to add new CPU support
  * Can be built as static tool or library

* Design and Implementation:

  * Extensive unit tests
  * Internal linker model can be dumped/read to textual format
  * Additional linking features can be plugged in as "passes"
  * OS specific and CPU specific code factored out

Why a new linker?
-----------------

The fact that clang relies on whatever linker tool you happen to have installed
means that clang has been very conservative adopting features which require a
recent linker.

In the same way that the MC layer of LLVM has removed clang's reliance on the
system assembler tool, the lld project will remove clang's reliance on the
system linker tool.


Current Status
--------------

lld can self host on x86-64 FreeBSD and Linux and x86 Windows.

All SingleSource tests in test-suite pass on x86-64 Linux.

All SingleSource and MultiSource tests in the LLVM test-suite
pass on MIPS 32-bit little-endian Linux.

Source
------

lld is available in the LLVM SVN repository::

  svn co http://llvm.org/svn/llvm-project/lld/trunk lld

lld is also available via the read-only git mirror::

  git clone http://llvm.org/git/lld.git

Put it in llvm's tools/ directory, rerun cmake, then build target lld.

Contents
--------

.. toctree::
   :maxdepth: 2

   design
   getting_started
   development
   windows_support
   open_projects
   sphinx_intro

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`

__ http://llvm.org/docs/DeveloperPolicy.html#license
