=======================================
Clang 7.0.0 (In-Progress) Release Notes
=======================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 7 release.
   Release notes for previous releases can be found on
   `the Download Page <http://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 7.0.0. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <http://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <http://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <http://clang.llvm.org>`_ or the
`LLVM Web Site <http://llvm.org>`_.

Note that if you are reading this file from a Subversion checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <http://llvm.org/releases/>`_.

What's New in Clang 7.0.0?
==========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- A new Implicit Conversion Sanitizer (``-fsanitize=implicit-conversion``) group
  was added. Please refer to the :ref:`release-notes-ubsan` section of the
  release notes for the details.

- Preliminary/experimental support for DWARF v5 debugging information. If you
  compile with ``-gdwarf-5 -O0`` you should get fully conforming DWARF v5
  information, including the new .debug_names accelerator table. Type units
  and split DWARF are known not to conform, and higher optimization levels
  will likely get a mix of v4 and v5 formats.
  
Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``-Wc++98-compat-extra-semi`` is a new flag, which was previously inseparable
  from ``-Wc++98-compat-pedantic``. The latter still controls the new flag.

- ``-Wextra-semi`` now also controls ``-Wc++98-compat-extra-semi``.
  Please do note that if you pass ``-Wno-c++98-compat-pedantic``, it implies
  ``-Wno-c++98-compat-extra-semi``, so if you want that diagnostic, you need
  to explicitly re-enable it (e.g. by appending ``-Wextra-semi``).

- ``-Wself-assign`` and ``-Wself-assign-field`` were extended to diagnose
  self-assignment operations using overloaded operators (i.e. classes).
  If you are doing such an assignment intentionally, e.g. in a unit test for
  a data structure, the first warning can be disabled by passing
  ``-Wno-self-assign-overloaded``, also the warning can be suppressed by adding
  ``*&`` to the right-hand side or casting it to the appropriate reference type.

Non-comprehensive list of changes in this release
-------------------------------------------------

- Clang binary and libraries have been renamed from 7.0 to 7.
  For example, the ``clang`` binary will be called ``clang-7``
  instead of ``clang-7.0``.

- Clang implements a collection of recent fixes to the C++ standard's definition
  of "standard-layout". In particular, a class is only considered to be
  standard-layout if all base classes and the first data member (or bit-field)
  can be laid out at offset zero.

- Clang's handling of the GCC ``packed`` class attribute in C++ has been fixed
  to apply only to non-static data members and not to base classes. This fixes
  an ABI difference between Clang and GCC, but creates an ABI difference between
  Clang 7 and earlier versions. The old behavior can be restored by setting
  ``-fclang-abi-compat`` to ``6`` or earlier.

- Clang implements the proposed resolution of LWG issue 2358, along with the
  `corresponding change to the Itanium C++ ABI
  <https://github.com/itanium-cxx-abi/cxx-abi/pull/51>`_, which make classes
  containing only unnamed non-zero-length bit-fields be considered non-empty.
  This is an ABI break compared to prior Clang releases, but makes Clang
  generate code that is ABI-compatible with other compilers. The old
  behavior can be restored by setting ``-fclang-abi-compat`` to ``6`` or
  lower.

- An existing tool named ``diagtool`` has been added to the release. As the
  name suggests, it helps with dealing with diagnostics in ``clang``, such as
  finding out the warning hierarchy, and which of them are enabled by default
  or for a particular compiler invocation.

- By default, Clang emits an address-significance table into
  every ELF object file when using the integrated assembler.
  Address-significance tables allow linkers to implement `safe ICF
  <https://research.google.com/pubs/archive/36912.pdf>`_ without the false
  positives that can result from other implementation techniques such as
  relocation scanning. The ``-faddrsig`` and ``-fno-addrsig`` flags can be
  used to control whether to emit the address-significance table.

- ...

New Compiler Flags
------------------

- ``-fstrict-float-cast-overflow`` and ``-fno-strict-float-cast-overflow``.

  When a floating-point value is not representable in a destination integer
  type, the code has undefined behavior according to the language standard. By
  default, Clang will not guarantee any particular result in that case. With the
  'no-strict' option, Clang attempts to match the overflowing behavior of the
  target's native float-to-int conversion instructions.

- ``-fforce-emit-vtables`` and ``-fno-force-emit-vtables``.

   In order to improve devirtualization, forces emitting of vtables even in
   modules where it isn't necessary. It causes more inline virtual functions
   to be emitted.

- ...

Deprecated Compiler Flags
-------------------------

The following options are deprecated and ignored. They will be removed in
future versions of Clang.

- ...

Modified Compiler Flags
-----------------------

- Before Clang 7, we prepended the `#` character to the `--autocomplete`
  argument to enable cc1 flags. For example, when the `-cc1` or `-Xclang` flag
  is in the :program:`clang` invocation, the shell executed
  `clang --autocomplete=#-<flag to be completed>`. Clang 7 now requires the
  whole invocation including all flags to be passed to the `--autocomplete` like
  this: `clang --autocomplete=-cc1,-xc++,-fsyn`.

New Pragmas in Clang
--------------------

Clang now supports the ...


Attribute Changes in Clang
--------------------------

- Clang now supports function multiversioning with attribute 'target' on ELF
  based x86/x86-64 environments by using indirect functions. This implementation
  has a few minor limitations over the GCC implementation for the sake of AST
  sanity, however it is otherwise compatible with existing code using this
  feature for GCC. Consult the documentation for the target attribute for more
  information.

- ...

Windows Support
---------------

- clang-cl's support for precompiled headers has been much improved:

   - When using a pch file, clang-cl now no longer redundantly emits inline
     methods that are already stored in the obj that was built together with
     the pch file (matching cl.exe).  This speeds up builds using pch files
     by around 30%.

   - The /Ycfoo.h and /Yufoo.h flags can now be used without /FIfoo.h when
     foo.h is instead included by an explicit `#include` directive. This means
     Visual Studio's default stdafx.h setup now uses precompiled headers with
     clang-cl.

- The alternative entry point names
  (``wmain``/``WinMain``/``wWinMain``/``DllMain``) now are properly mangled
  as plain C names in C++ contexts when targeting MinGW, without having to
  explicit specify ``extern "C"``. (This was already the case for MSVC
  targets.)

- ...


C Language Changes in Clang
---------------------------

- ...

...

C11 Feature Support
^^^^^^^^^^^^^^^^^^^

...

C++ Language Changes in Clang
-----------------------------

- ...

C++1z Feature Support
^^^^^^^^^^^^^^^^^^^^^

...

Objective-C Language Changes in Clang
-------------------------------------

...

OpenCL C Language Changes in Clang
----------------------------------

...

OpenMP Support in Clang
----------------------------------

- Clang gained basic support for OpenMP 4.5 offloading for NVPTX target.
   To compile your program for NVPTX target use the following options:
   `-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda` for 64 bit platforms or
   `-fopenmp -fopenmp-targets=nvptx-nvidia-cuda` for 32 bit platform.

- Passing options to the OpenMP device offloading toolchain can be done using
  the `-Xopenmp-target=<triple> -opt=val` flag. In this way the `-opt=val`
  option will be forwarded to the respective OpenMP device offloading toolchain
  described by the triple. For example passing the compute capability to
  the OpenMP NVPTX offloading toolchain can be done as follows:
  `-Xopenmp-target=nvptx64-nvidia-cuda -march=sm_60`. For the case when only one
  target offload toolchain is specified under the `-fopenmp-targets=<triples>`
  option, then the triple can be skipped: `-Xopenmp-target -march=sm_60`.

- Other bugfixes.

CUDA Support in Clang
---------------------

- Clang will now try to locate the CUDA installation next to :program:`ptxas`
  in the `PATH` environment variable. This behavior can be turned off by passing
  the new flag `--cuda-path-ignore-env`.

- Clang now supports generating object files with relocatable device code. This
  feature needs to be enabled with `-fcuda-rdc` and my result in performance
  penalties compared to whole program compilation. Please note that NVIDIA's
  :program:`nvcc` must be used for linking.

Internal API Changes
--------------------

These are major API changes that have happened since the 6.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

-  ...

AST Matchers
------------

- ...

clang-format
------------

- Clang-format will now support detecting and formatting code snippets in raw
  string literals.  This is configured through the `RawStringFormats` style
  option.

- ...

libclang
--------

...


Static Analyzer
---------------

- The new `MmapWriteExec` checker had been introduced to detect attempts to map pages
both writable and executable.

...

.. _release-notes-ubsan:

Undefined Behavior Sanitizer (UBSan)
------------------------------------

* A new Implicit Conversion Sanitizer (``-fsanitize=implicit-conversion``) group
  was added.

  Currently, only one type of issues is caught - implicit integer truncation
  (``-fsanitize=implicit-integer-truncation``), also known as integer demotion.
  While there is a ``-Wconversion`` diagnostic group that catches this kind of
  issues, it is both noisy, and does not catch **all** the cases.

  .. code-block:: c++

      unsigned char store = 0;

      bool consume(unsigned int val);

      void test(unsigned long val) {
        if (consume(val)) // the value may have been silently truncated.
          store = store + 768; // before addition, 'store' was promoted to int.
        (void)consume((unsigned int)val); // OK, the truncation is explicit.
      }

  Just like other ``-fsanitize=integer`` checks, these issues are **not**
  undefined behaviour. But they are not *always* intentional, and are somewhat
  hard to track down. This group is **not** enabled by ``-fsanitize=undefined``,
  but the ``-fsanitize=implicit-integer-truncation`` check
  is enabled by ``-fsanitize=integer``.

Core Analysis Improvements
==========================

- ...

New Issues Found
================

- ...

Python Binding Changes
----------------------

The following methods have been added:

-  ...

Significant Known Problems
==========================

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <http://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Subversion version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us via the `mailing
list <http://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
