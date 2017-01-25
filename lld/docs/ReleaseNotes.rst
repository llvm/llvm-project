=======================
LLD 4.0.0 Release Notes
=======================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 4.0.0 release.

Introduction
============

LLD is a linker which supports ELF (Unix), COFF (Windows) and Mach-O
(macOS). It is generally faster than the GNU BFD/gold linkers or the
MSVC linker.

LLD is designed to be a drop-in replacmenet for the system linkers, so
that users don't need to change their build systems other than swapping
the linker command.

This document contains the release notes for LLD 4.0.0.
Here we describe the status of LLD, including major improvements
from the previous release. All LLD releases may be downloaded
from the `LLVM releases web site <http://llvm.org/releases/>`_.


What's New in LLD 4.0?
======================

ELF Improvements
----------------

LLD provides much better compatibility with the GNU linker than before.
Now it is able to link the entire FreeBSD base system including the kernel
out of the box. We are working closely with the FreeBSD project to
make it usable as the system linker in a future release of the operating
system.

Multi-threading performance has been improved, and multi-threading
is now enabled by default. Combined with other optimizations, LLD 4.0
is about 1.5 times faster than LLD 3.9 when linking large programs
in our test environment.

Other notable changes are listed below:

* Error messages contain more information than before. If debug info
  is available, the linker prints out not only the object file name
  but the source location of unresolved symbols.

* Error messages are printed in red just like Clang by default. You
  can disable it by passing -no-color-diagnostics.

* LLD's version string is now embedded in a .comment section in the
  result output file. You can dump it with this command: ``objdump -j -s
  .comment <file>``.

* The -Map option is supported. With that, you can print out section
  and symbol information to a specified file. This feature is useful
  for analyzing link results.

* The file format for the -reproduce option has changed from cpio to
  tar.

* When creating a copy relocation for a symbol, LLD now scans the
  DSO's header to see if the symbol is in a read-only segment. If so,
  space for the copy relocation is reserved in .bss.rel.ro instead of
  .bss. This fixes a security issue that read-only data in a DSO
  becomes writable if it is copied by a copy relocation. This issue
  was disclosed originally on the binutils mailing list at
  `<https://sourceware.org/ml/libc-alpha/2016-12/msg00914.html>`.

* Default image base address for x86-64 has been changed from 0x10000
  to 0x200000 to make it huge-page friendly.

* Compressed input sections are supported.

* ``--oformat binary``, ``--section-start``, ``-Tbss``, ``-Tdata``,
  ``-Ttext``, ``-b binary``, ``-build-id=uuid``, ``-no-rosegment``,
  ``-nopie``, ``-nostdlib``, ``-omagic``, ``-retain-symbols-file``,
  ``-sort-section``, ``-z max-page-size`` and ``-z wxneeded`` are
  suppoorted.

* A lot of linker script directives have been added.

COFF Improvements
-----------------

* Performance on Windows has been improved by parallelizing parts of the
  linker and optimizing file system operations. As a result of these
  improvements, LLD 4.0 has been measured to be about 2.5 times faster
  than LLD 3.9 when linking a large Chromium DLL.

