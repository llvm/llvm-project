=====================================
Clang 3.7 (In-Progress) Release Notes
=====================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 3.7 release. You may
   prefer the `Clang 3.6 Release Notes
   <http://llvm.org/releases/3.6.0/tools/clang/docs/ReleaseNotes.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 3.7. Here we
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

What's New in Clang 3.7?
========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Use of the ``__declspec`` language extension for declaration attributes now
  requires passing the -fms-extensions or -fborland compiler flag. This language
  extension is also enabled when compiling CUDA code, but its use should be
  viewed as an implementation detail that is subject to change.

- Clang 3.7 fully supports OpenMP 3.1 and reported to work on many platforms,
  including x86, x86-64 and Power. Also, pragma ``omp simd`` from OpenMP 4.0 is
  supported as well. See below for details.


Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang's diagnostics are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate source information
about them. The improvements since the 3.6 release include:

- -Wrange-loop-analysis analyzes the loop variable type and the container type
  to determine whether copies are made of the container elements.  If possible,
  suggest a const reference type to prevent copies, or a non-reference type
  to indicate a copy is made.

- -Wredundant-move warns when a parameter variable is moved on return and the
  return type is the same as the variable.  Returning the variable directly
  will already make a move, so the call is not needed.

- -Wpessimizing-move warns when a local variable is ir moved on return and the
  return type is the same as the variable.  Copy elision cannot take place with
  a move, but can take place if the variable is returned directly.

- -Wmove is a new warning group which has the previous two warnings,
  -Wredundant-move and -Wpessimizing-move, as well as previous warning
  -Wself-move.  In addition, this group is part of -Wmost and -Wall now.

- -Winfinite-recursion, a warning for functions that only call themselves,
  is now part of -Wmost and -Wall.

New Compiler Flags
------------------

The sized deallocation feature of C++14 is now controlled by the
``-fsized-deallocation`` flag. This feature relies on library support that
isn't yet widely deployed, so the user must supply an extra flag to get the
extra functionality.

The option ....


New Pragmas in Clang
-----------------------

Clang now supports the ...

Windows Support
---------------

Clang's support for building native Windows programs ...


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

OpenMP Support
--------------
OpenMP 3.1 is fully supported, but disabled by default. To enable it, please use
``-fopenmp=libomp`` command line option. Your feedback (positive or negative) on
using OpenMP-enabled clang would be much appreciated; please share it either on
`cfe-dev <http://lists.cs.uiuc.edu/mailman/listinfo/cfe-dev>`_ or `openmp-dev
<http://lists.cs.uiuc.edu/mailman/listinfo/openmp-dev>`_ mailing lists.

In addition to OpenMP 3.1, several important elements of 4.0 version of the
standard are supported as well:
- ``omp simd``, ``omp for simd`` and ``omp parallel for simd`` pragmas
- atomic constructs
- ``proc_bind`` clause of ``omp parallel`` pragma
- ``depend`` clause of ``omp task`` pragma (except for array sections)
- ``omp cancel`` and ``omp cancellation point`` pragmas
- ``omp taskgroup`` pragma
...

Internal API Changes
--------------------

These are major API changes that have happened since the 3.6 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

-  Some of the `PPCallbacks` interface now deals in `MacroDefinition`
   objects instead of `MacroDirective` objects. This allows preserving
   full information on macros imported from modules.

-  `clang-c/Index.h` no longer `#include`\s `clang-c/Documentation.h`.
   You now need to explicitly `#include "clang-c/Documentation.h"` if
   you use the libclang documentation API.

libclang
--------

...

Static Analyzer
---------------

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
list <http://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
