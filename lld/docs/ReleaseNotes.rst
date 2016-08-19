======================
LLD 3.9 Release Notes
======================

.. contents::
    :local:

Introduction
============

This document contains the release notes for the LLD linker, release 3.9.
Here we describe the status of LLD, including major improvements
from the previous release. All LLD releases may be downloaded
from the `LLVM releases web site <http://llvm.org/releases/>`_.

What's new in ELF Support?
==========================

LLD 3.9 is a major milestone for us. It is the first release that can
link real-world large userland programs, including LLVM/Clang/LLD
themselves. In fact, for example, it can now be used to produce most
userland programs distributed as part of FreeBSD.

Many contributors have joined to the project to develop new features,
port it to new architectures and fix issues since the last release.

Link-Time Optimization
----------------------

Initial support for LTO has been added. It is compatible with
`the LLVM gold plugin <http://llvm.org/docs/GoldPlugin.html>`_ in terms of
command line flags and input file format so that LLD is usable as a
drop-in replacement for GNU gold. LTO is implemented as a native
feature unlike the GNU gold's plugin mechanism.

Identical Code Folding
----------------------

LLD 3.9 can now merge identical code sections to produce smaller
output files. It is expected to be used with ``-ffunction-sections``.

Symbol Versioning
-----------------

LLD 3.9 is able to link against versioned symbols as well as produce
versioned symbols. Both the original Sun's symbol versioning scheme
and the GNU extension are supported.

New Targets
-----------

LLD has expanded support for new targets, including ARM/Thumb, the x32
ABI and MIPS N64 ABI, in addition to the existing support for x86,
x86-64, MIPS, PowerPC and PPC64.

TLS Relocation Optimizations
----------------------------

The ELF ABI specification of the thread-local variable define a few
peephole optimizations linkers can do by rewriting instructions at the
link-time to reduce run-time overhead to access TLS variables. That
feature has been implemented.

New Linker Flags
----------------

Many command line options have been added in this release, including:

- Symbol resolution and output options: ``-Bsymbolic-functions``,
  ``-export-dynamic-symbol``, ``-image-base``, ``-pie``, ``-end-lib``,
  ``-start-lib``, ``-build-id={md5,sha1,none,0x<hexstring>}``.

- Symbol versioning option: ``-dynamic-list``.

- LTO options: ``-lto-O``, ``-lto-aa-pipeline``, ``-lto-jobs``,
  ``-lto-newpm-passes``, ``-plugin``, ``-plugin-eq``, ``-plugin-opt``,
  ``-plugin-opt-eq``, ``-disable-verify``, ``-mllvm``.

- Driver optionss: ``-help``, ``-version``, ``-unresolved-symbols``.

- Debug options: ``-demangle``, ``-reproduce``, ``-save-temps``,
  ``-strip-debug``, ``-trace``, ``-trace-symbol``,
  ``-warn-execstack``.

- Exception handling option: ``-eh-frame-hdr``.

- Identical Code Folding option: ``-icf``.

Changes to the MIPS Target
--------------------------

* Added support for MIPS N64 ABI.
* Added support for TLS relocations for both O32 and N64 MIPS ABIs.

Building LLVM Toolchain with LLD
--------------------------------

A new CMake variable, ``LLVM_ENABLE_LLD``, has been added to use LLD
to build the LLVM toolchain. If the varaible is true, ``-fuse-ld=lld``
option will be added to linker flags so that ``ld.lld`` is used
instead of default ``ld``.  Because ``-fuse-ld=lld`` is a new compiler
driver option, you need Clang 3.8 or newer to use the feature.
