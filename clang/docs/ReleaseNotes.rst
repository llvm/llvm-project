========================================
Clang 14.0.0 (In-Progress) Release Notes
========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 14 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 14.0.0. Here we
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

What's New in Clang 14.0.0?
===========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

-  ...

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- -Wbitwise-instead-of-logical (part of -Wbool-operation) warns about use of bitwise operators with boolean operands which have side effects.

Non-comprehensive list of changes in this release
-------------------------------------------------

- Maximum _ExtInt size was decreased from 16,777,215 bits to 8,388,608 bits.
  Motivation for this was discussed in PR51829.

New Compiler Flags
------------------

- ...

Deprecated Compiler Flags
-------------------------

- ...

Modified Compiler Flags
-----------------------

- Support has been added for the following processors (``-mcpu`` identifiers in parentheses):

  - RISC-V SiFive E20 (``sifive-e20``).
  - RISC-V SiFive E21 (``sifive-e21``).
  - RISC-V SiFive E24 (``sifive-e24``).
  - RISC-V SiFive E34 (``sifive-e34``).
  - RISC-V SiFive S21 (``sifive-s21``).
  - RISC-V SiFive S51 (``sifive-s51``).
  - RISC-V SiFive S54 (``sifive-s54``).
  - RISC-V SiFive S76 (``sifive-s76``).

- Support has been added for the following architectures (``-march`` identifiers in parentheses):

  - Armv9-A (``armv9-a``).
  - Armv9.1-A (``armv9.1-a``).
  - Armv9.2-A (``armv9.2-a``).

Removed Compiler Flags
-------------------------

- ...

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- Attributes loaded as clang plugins which are sensitive to LangOpts must
  now override ``acceptsLangOpts`` instead of ``diagLangOpts``.
  Returning false will produce a generic "attribute ignored" diagnostic, as
  with clang's built-in attributes.
  If plugins want to provide richer diagnostics, they can do so when the
  attribute is handled instead, e.g. in ``handleDeclAttribute``.
  (This was changed in order to better support attributes in code completion).

Windows Support
---------------

- An MSVC compatibility workaround for C++ operator names was removed. As a
  result, the ``<query.h>`` Windows SDK header may not compile out of the box.
  Users should use a recent SDK and pass ``-DQUERY_H_RESTRICTION_PERMISSIVE``
  or pass ``/permissive`` to disable C++ operator names altogether. See
  `PR42427 <https://llvm.org/pr42427>` for more info.

C Language Changes in Clang
---------------------------

- Wide multi-characters literals such as ``L'ab'`` that would previously be interpreted as ``L'b'``
  are now ill-formed in all language modes. The motivation for this change is outlined in
  `P2362 <wg21.link/P2362>`_.
- Support for ``__attribute__((error("")))`` and
  ``__attribute__((warning("")))`` function attributes have been added.
- The maximum allowed alignment has been increased from 2^29 to 2^32.

C++ Language Changes in Clang
-----------------------------

- ...

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^
...

C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Implemented `P1938R3: if consteval <https://wg21.link/P1938R3>`_.
- Implemented `P2360R0: Extend init-statement to allow alias-declaration <https://wg21.link/P2360R0>`_.


CUDA Language Changes in Clang
------------------------------

- Clang now supports CUDA versions up to 11.4.
- Default GPU architecture has been changed from sm_20 to sm_35.

Objective-C Language Changes in Clang
-------------------------------------

OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------

OpenMP Support in Clang
-----------------------

- ``clang-nvlink-wrapper`` tool introduced to support linking of cubin files archived in an archive. See :doc:`ClangNvlinkWrapper`.


CUDA Support in Clang
---------------------

- ...

X86 Support in Clang
--------------------

- Support for ``AVX512-FP16`` instructions has been added.

Internal API Changes
--------------------

- ...

Build System Changes
--------------------

- ...

AST Matchers
------------

- ``TypeLoc`` AST Matchers are now available. These matchers provide helpful
  utilities for matching ``TypeLoc`` nodes, such as the ``pointerTypeLoc``
  matcher or the ``hasReturnTypeLoc`` matcher. The addition of these matchers
  was made possible by changes to the handling of ``TypeLoc`` nodes that
  allows them to enjoy the same static type checking as other AST node kinds.

clang-format
------------

- Option ``AllowShortEnumsOnASingleLine: false`` has been improved, it now
  correctly places the opening brace according to ``BraceWrapping.AfterEnum``.

- Option ``QualifierAligment`` has been added in order to auto-arrange the
  positioning of specifiers/qualifiers
  `const` `volatile` `static` `inline` `constexpr` `restrict`
  in variable and parameter declarations to be either ``Right`` aligned
  or ``Left`` aligned or ``Custom`` using ``QualifierOrder``.

- Option ``QualifierOrder`` has been added to allow the order
  `const` `volatile` `static` `inline` `constexpr` `restrict`
  to be controlled relative to the `type`.

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
