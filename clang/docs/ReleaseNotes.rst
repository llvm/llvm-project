=====================================
Clang 3.5 (In-Progress) Release Notes
=====================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 3.5 release. You may
   prefer the `Clang 3.4 Release Notes
   <http://llvm.org/releases/3.4/tools/clang/docs/ReleaseNotes.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 3.5. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <http://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <http://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please check out the main please see the `Clang Web
Site <http://clang.llvm.org>`_ or the `LLVM Web
Site <http://llvm.org>`_.

Note that if you are reading this file from a Subversion checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <http://llvm.org/releases/>`_.

What's New in Clang 3.5?
========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Clang uses the new MingW ABI
  GCC 4.7 changed the mingw ABI. Clang 3.4 and older use the GCC 4.6
  ABI. Clang 3.5 and newer use the GCC 4.7 abi.

- The __has_attribute feature test is now target-aware. Older versions of Clang
  would return true when the attribute spelling was known, regardless of whether
  the attribute was available to the specific target. Clang now returns true
  only when the attribute pertains to the current compilation target.


Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang's diagnostics are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate source information
about them. The improvements since the 3.4 release include:

- GCC compatibility: Clang displays a warning on unsupported gcc
  optimization flags instead of an error.

-  ...

New Compiler Flags
------------------

The integrated assembler is now turned on by default on ARM (and Thumb),
so the use of the option `-fintegrated-as` is now redundant on those
architectures. This is an important move to both *eat our own dog food*
and to ease cross-compilation tremendously.

We are aware of the problems that this may cause for code bases that
rely on specific GNU syntax or extensions, and we're working towards
getting them all fixed. Please, report bugs or feature requests if
you find anything. In the meantime, use `-fno-integrated-as` to revert
back the call to GNU assembler.

In order to provide better diagnostics, the integrated assembler validates
inline assembly when the integrated assembler is enabled.  Because this is
considered a feature of the compiler, it is controlled via the `fintegrated-as`
and `fno-integrated-as` flags which enable and disable the integrated assembler
respectively.  `-integrated-as` and `-no-integrated-as` are now considered
legacy flags (but are available as an alias to prevent breaking existing users),
and users are encouraged to switch to the equivalent new feature flag.

Deprecated flags `-faddress-sanitizer`, `-fthread-sanitizer`,
`-fcatch-undefined-behavior` and `-fbounds-checking` were removed in favor of
`-fsanitize=` family of flags.

It is now possible to get optimization reports from the major transformation
passes via three new flags: `-Rpass`, `-Rpass-missed` and `-Rpass-analysis`.
These flags take a POSIX regular expression which indicates the name
of the pass (or passes) that should emit optimization remarks.

Options `-u` and `-z` are forwarded to the linker on gnutools toolchains.


New Pragmas in Clang
-----------------------

Loop optimization hints can be specified using the new `#pragma clang loop`
directive just prior to the desired loop. The directive allows vectorization,
interleaving, and unrolling to be enabled or disabled. Vector width as well
as interleave and unrolling count can be manually specified.  See language
extensions for details.

Clang now supports the `#pragma unroll` directive to specify loop unrolling
optimization hints.  Placed just prior to the desired loop, `#pragma unroll`
directs the loop unroller to attempt to fully unroll the loop.  The pragma may
also be specified with a positive integer parameter indicating the desired
unroll count: `#pragma unroll _value_`.  The unroll count parameter can be
optionally enclosed in parentheses.

C Language Changes in Clang
---------------------------

...

C11 Feature Support
^^^^^^^^^^^^^^^^^^^

...

C++ Language Changes in Clang
-----------------------------

- ...

C++11 Feature Support
^^^^^^^^^^^^^^^^^^^^^

...

Objective-C Language Changes in Clang
-------------------------------------

...

OpenCL C Language Changes in Clang
----------------------------------

...

Internal API Changes
--------------------

These are major API changes that have happened since the 3.4 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

...

libclang
--------

...

Static Analyzer
---------------

The `-analyzer-config` options are now passed from scan-build through to
ccc-analyzer and then to Clang.

With the option `-analyzer-config stable-report-filename=true`,
instead of `report-XXXXXX.html`, scan-build/clang analyzer generate
`report-<filename>-<function, method name>-<function position>-<id>.html`.
(id = i++ for several issues found in the same function/method).

List the function/method name in the index page of scan-build.

...

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
list <http://lists.cs.uiuc.edu/mailman/listinfo/cfe-dev>`_.
