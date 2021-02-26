========================================
Clang 13.0.0 (In-Progress) Release Notes
========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 13 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 13.0.0. Here we
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

What's New in Clang 13.0.0?
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

- ...

Non-comprehensive list of changes in this release
-------------------------------------------------

- ...

New Compiler Flags
------------------

- ...

Deprecated Compiler Flags
-------------------------

- ...

Modified Compiler Flags
-----------------------

- -Wshadow now also checks for shadowed structured bindings

Removed Compiler Flags
-------------------------

- The clang-cl ``/fallback`` flag, which made clang-cl invoke Microsoft Visual
  C++ on files it couldn't compile itself, has been removed.

- ``-Wreturn-std-move-in-c++11``, which checked whether an entity is affected by
  `CWG1579 <https://wg21.link/CWG1579>`_ to become implicitly movable, has been
  removed.

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

C++ Language Changes in Clang
-----------------------------

- ...

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

X86 Support in Clang
--------------------

- ...

Internal API Changes
--------------------

These are major API changes that have happened since the 12.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

- ...

Build System Changes
--------------------

These are major changes to the build system that have happened since the 12.0.0
release of Clang. Users of the build system should adjust accordingly.

- ...

AST Matchers
------------

- ...

clang-format
------------

- Option ``SpacesInLineCommentPrefix`` has been added to control the
  number of spaces in a line comments prefix.

- Option ``SortIncludes`` has been updated from a ``bool`` to an
  ``enum`` with backwards compatibility. In addition to the previous
  ``true``/``false`` states (now ``CaseInsensitive``/``Never``), a third
  state has been added (``CaseSensitive``) which causes an alphabetical sort
  with case used as a tie-breaker.

  .. code-block:: c++

    // Never (previously false)
    #include "B/A.h"
    #include "A/B.h"
    #include "a/b.h"
    #include "A/b.h"
    #include "B/a.h"

    // CaseInsensitive (previously true)
    #include "A/B.h"
    #include "A/b.h"
    #include "B/A.h"
    #include "B/a.h"
    #include "a/b.h"

    // CaseSensitive
    #include "A/B.h"
    #include "A/b.h"
    #include "a/b.h"
    #include "B/A.h"
    #include "B/a.h"

- ``BasedOnStyle: InheritParentConfig`` allows to use the ``.clang-format`` of
  the parent directories to overwrite only parts of it.

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
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us via the `mailing
list <https://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
