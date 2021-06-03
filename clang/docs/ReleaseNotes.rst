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

- Guaranteed tail calls are now supported with statement attributes
  ``[[clang::musttail]]`` in C++ and ``__attribute__((musttail))`` in C. The
  attribute is applied to a return statement (not a function declaration),
  and an error is emitted if a tail call cannot be guaranteed, for example if
  the function signatures of caller and callee are not compatible. Guaranteed
  tail calls enable a class of algorithms that would otherwise use an
  arbitrary amount of stack space.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ...

Non-comprehensive list of changes in this release
-------------------------------------------------

- The default value of _MSC_VER was raised from 1911 to 1914. MSVC 19.14 has the
  support to overaligned objects on x86_32 which is required for some LLVM 
  passes.

New Compiler Flags
------------------

- ``-Wreserved-identifier`` emits warning when user code uses reserved
  identifiers.

- ``-fstack-usage`` generates an extra .su file per input source file. The .su
  file contains frame size information for each function defined in the source
  file.

Deprecated Compiler Flags
-------------------------

- ...

Modified Compiler Flags
-----------------------

- -Wshadow now also checks for shadowed structured bindings
- ``-B <prefix>`` (when ``<prefix>`` is a directory) was overloaded to additionally
  detect GCC installations under ``<prefix>`` (``lib{,32,64}/gcc{,-cross}/$triple``).
  This behavior was incompatible with GCC, caused interop issues with
  ``--gcc-toolchain``, and was thus dropped. Specify ``--gcc-toolchain=<dir>``
  instead. ``-B``'s other GCC-compatible semantics are preserved:
  ``$prefix/$triple-$file`` and ``$prefix$file`` are searched for executables,
  libraries, includes, and data files used by the compiler.

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

- Added support for C++11-style ``[[]]`` attributes on using-declarations, as a
  clang extension.

Windows Support
---------------

C Language Changes in Clang
---------------------------

- ...

C++ Language Changes in Clang
-----------------------------

- The oldest supported GNU libstdc++ is now 4.8.3 (released 2014-05-22).
  Clang workarounds for bugs in earlier versions have been removed.

- ...

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^
...

C++2b Feature Support
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

- The option ``LIBCLANG_INCLUDE_CLANG_TOOLS_EXTRA`` no longer exists. There were
  two releases with that flag forced off, and no uses were added that forced it
  on. The recommended replacement is clangd.

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
  ``true``/``false`` states (now ``CaseSensitive``/``Never``), a third
  state has been added (``CaseInsensitive``) which causes an alphabetical sort
  with case used as a tie-breaker.

  .. code-block:: c++

    // Never (previously false)
    #include "B/A.h"
    #include "A/B.h"
    #include "a/b.h"
    #include "A/b.h"
    #include "B/a.h"

    // CaseSensitive (previously true)
    #include "A/B.h"
    #include "A/b.h"
    #include "B/A.h"
    #include "B/a.h"
    #include "a/b.h"

    // CaseInsensitive
    #include "A/B.h"
    #include "A/b.h"
    #include "a/b.h"
    #include "B/A.h"
    #include "B/a.h"

- ``BasedOnStyle: InheritParentConfig`` allows to use the ``.clang-format`` of
  the parent directories to overwrite only parts of it.

- Option ``IndentAccessModifiers`` has been added to be able to give access
  modifiers their own indentation level inside records.

- Option ``PPIndentWidth`` has been added to be able to configure pre-processor
  indentation independent from regular code.

- Option ``ShortNamespaceLines`` has been added to give better control
  over ``FixNamespaceComments`` when determining a namespace length.

- Support for Whitesmiths has been improved, with fixes for ``namespace`` blocks
  and ``case`` blocks and labels.

- Option ``EmptyLineAfterAccessModifier`` has been added to remove, force or keep
  new lines after access modifiers.

- Checks for newlines in option ``EmptyLineBeforeAccessModifier`` are now based
  on the formatted new lines and not on the new lines in the file. (Fixes
  https://llvm.org/PR41870.)

- Option ``SpacesInAngles`` has been improved, it now accepts ``Leave`` value
  that allows to keep spaces where they are already present.

- Option ``AllowShortIfStatementsOnASingleLine`` has been improved, it now
  accepts ``AllIfsAndElse`` value that allows to put "else if" and "else" short
  statements on a single line. (Fixes https://llvm.org/PR50019.)

- Option ``BreakInheritanceList`` gets a new style, ``AfterComma``. It breaks
  only after the commas that separate the base-specifiers.

- ``git-clang-format`` no longer formats changes to symbolic links. (Fixes
  https://llvm.org/PR46992.)

- Makes ``PointerAligment: Right`` working with ``AlignConsecutiveDeclarations``.
  (Fixes https://llvm.org/PR27353)

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
