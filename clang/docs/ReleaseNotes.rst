========================================
Clang 11.0.0 (In-Progress) Release Notes
========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 11 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 11.0.0. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Clang 11.0.0?
===========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- ...

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- -Wpointer-to-int-cast is a new warning group. This group warns about C-style
  casts of pointers to a integer type too small to hold all possible values.

Non-comprehensive list of changes in this release
-------------------------------------------------


New Compiler Flags
------------------


- -fstack-clash-protection will provide a protection against the stack clash
  attack for x86 architecture through automatic probing of each page of
  allocated stack.

Deprecated Compiler Flags
-------------------------

The following options are deprecated and ignored. They will be removed in
future versions of Clang.

- ...

Modified Compiler Flags
-----------------------


New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- ...

Windows Support
---------------

C Language Changes in Clang
---------------------------

- ...

C11 Feature Support
^^^^^^^^^^^^^^^^^^^

...

C++ Language Changes in Clang
-----------------------------

- Clang now implements a restriction on giving non-C-compatible anonymous
  structs a typedef name for linkage purposes, as described in C++ committee
  paper `P1766R1 <http://wg21.link/p1766r1>`. This paper was adopted by the
  C++ committee as a Defect Report resolution, so it is applied retroactively
  to all C++ standard versions. This affects code such as:

  .. code-block:: c++

    typedef struct {
      int f() { return 0; }
    } S;

  Previous versions of Clang rejected some constructs of this form
  (specifically, where the linkage of the type happened to be computed
  before the parser reached the typedef name); those cases are still rejected
  in Clang 11.  In addition, cases that previous versions of Clang did not
  reject now produce an extension warning. This warning can be disabled with
  the warning flag ``-Wno-non-c-typedef-for-linkage``.

  Affected code should be updated to provide a tag name for the anonymous
  struct:

  .. code-block:: c++

    struct S {
      int f() { return 0; }
    };

  If the code is shared with a C compilation (for example, if the parts that
  are not C-compatible are guarded with ``#ifdef __cplusplus``), the typedef
  declaration should be retained, but a tag name should still be provided:

  .. code-block:: c++

    typedef struct S {
      int f() { return 0; }
    } S;

C++1z Feature Support
^^^^^^^^^^^^^^^^^^^^^

...

Objective-C Language Changes in Clang
-------------------------------------


OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------


OpenMP Support in Clang
-----------------------

- ...

CUDA Support in Clang
---------------------

- ...

Internal API Changes
--------------------

These are major API changes that have happened since the 10.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.


Build System Changes
--------------------

These are major changes to the build system that have happened since the 10.0.0
release of Clang. Users of the build system should adjust accordingly.

- ...

AST Matchers
------------

- ...

clang-format
------------


- Option ``IndentCaseBlocks`` has been added to support treating the block
  following a switch case label as a scope block which gets indented itself.
  It helps avoid having the closing bracket align with the switch statement's
  closing bracket (when ``IndentCaseLabels`` is ``false``).

- Option ``ObjCBreakBeforeNestedBlockParam`` has been added to optionally apply
  linebreaks for function arguments declarations before nested blocks.

  .. code-block:: c++

    switch (fool) {                vs.     switch (fool) {
    case 1:                                case 1: {
      {                                      bar();
         bar();                            } break;
      }                                    default: {
      break;                                 plop();
    default:                               }
      {                                    }
        plop();
      }
    }

- Option ``InsertTrailingCommas`` can be set to ``TCS_Wrapped`` to insert
  trailing commas in container literals (arrays and objects) that wrap across
  multiple lines. It is currently only available for JavaScript and disabled by
  default (``TCS_None``).

- Option ``BraceWrapping.BeforeLambdaBody`` has been added to manage lambda
  line break inside function parameter call in Allman style.

  .. code-block:: c++

      true:
      connect(
        []()
        {
          foo();
          bar();
        });

      false:
      connect([]() {
          foo();
          bar();
        });

libclang
--------

- ...

Static Analyzer
---------------

- ...

.. _release-notes-ubsan:

Undefined Behavior Sanitizer (UBSan)
------------------------------------


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
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Subversion version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us via the `mailing
list <https://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
