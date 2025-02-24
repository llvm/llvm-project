===========================================
Clang |release| |ReleaseNotesTitle|
===========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming Clang |version| release.
     Release notes for previous releases can be found on
     `the Releases Page <https://llvm.org/releases/>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release |release|. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. For the libc++ release notes,
see `this page <https://libcxx.llvm.org/ReleaseNotes.html>`_. All LLVM releases
may be downloaded from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Potentially Breaking Changes
============================

- The Objective-C ARC migrator (ARCMigrate) has been removed.

C/C++ Language Potentially Breaking Changes
-------------------------------------------

C++ Specific Potentially Breaking Changes
-----------------------------------------

- The type trait builtin ``__is_referenceable`` has been removed, since it has
  very few users and all the type traits that could benefit from it in the
  standard library already have their own bespoke builtins.

ABI Changes in This Version
---------------------------

- Return larger CXX records in memory instead of using AVX registers. Code compiled with older clang will be incompatible with newer version of the clang unless -fclang-abi-compat=20 is provided. (#GH120670)

AST Dumping Potentially Breaking Changes
----------------------------------------

Clang Frontend Potentially Breaking Changes
-------------------------------------------

Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------

What's New in Clang |release|?
==============================

C++ Language Changes
--------------------

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++17 Feature Support
^^^^^^^^^^^^^^^^^^^^^

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The flag `-frelaxed-template-template-args`
  and its negation have been removed, having been deprecated since the previous
  two releases. The improvements to template template parameter matching implemented
  in the previous release, as described in P3310 and P3579, made this flag unnecessary.

C Language Changes
------------------

- Clang now allows an ``inline`` specifier on a typedef declaration of a
  function type in Microsoft compatibility mode. #GH124869

C2y Feature Support
^^^^^^^^^^^^^^^^^^^

C23 Feature Support
^^^^^^^^^^^^^^^^^^^

Non-comprehensive list of changes in this release
-------------------------------------------------

New Compiler Flags
------------------

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
-------------------------

Attribute Changes in Clang
--------------------------

- The ``no_sanitize`` attribute now accepts both ``gnu`` and ``clang`` names.

Improvements to Clang's diagnostics
-----------------------------------

- Improve the diagnostics for deleted default constructor errors for C++ class
  initializer lists that don't explicitly list a class member and thus attempt
  to implicitly default construct that member.
- The ``-Wunique-object-duplication`` warning has been added to warn about objects
  which are supposed to only exist once per program, but may get duplicated when
  built into a shared library.
- Fixed a bug where Clang's Analysis did not correctly model the destructor behavior of ``union`` members (#GH119415).

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The behvaiour of ``__add_pointer`` and ``__remove_pointer`` for Objective-C++'s ``id`` and interfaces has been fixed.

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 - Fixed crash when a parameter to the ``clang::annotate`` attribute evaluates to ``void``. See #GH119125

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Clang is now better at keeping track of friend function template instance contexts. (#GH55509)
- The initialization kind of elements of structured bindings
  direct-list-initialized from an array is corrected to direct-initialization.

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

- HTML tags in comments that span multiple lines are now parsed correctly by Clang's comment parser. (#GH120843)

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

NVPTX Support
^^^^^^^^^^^^^^

Hexagon Support
^^^^^^^^^^^^^^^

-  The default compilation target has been changed from V60 to V68.

X86 Support
^^^^^^^^^^^

- Disable ``-m[no-]avx10.1`` and switch ``-m[no-]avx10.2`` to alias of 512 bit
  options.
- Change ``-mno-avx10.1-512`` to alias of ``-mno-avx10.1-256`` to disable both
  256 and 512 bit instructions.

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

-  Support for __ptrauth type qualifier has been added.

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA Support
^^^^^^^^^^^^

AIX Support
^^^^^^^^^^^

NetBSD Support
^^^^^^^^^^^^^^

WebAssembly Support
^^^^^^^^^^^^^^^^^^^

AVR Support
^^^^^^^^^^^

DWARF Support in Clang
----------------------

Floating Point Support in Clang
-------------------------------

Fixed Point Support in Clang
----------------------------

AST Matchers
------------

clang-format
------------

- Adds ``BreakBeforeTemplateCloser`` option.
- Adds ``BinPackLongBracedList`` option to override bin packing options in
  long (20 item or more) braced list initializer lists.

libclang
--------

Code Completion
---------------

Static Analyzer
---------------

New features
^^^^^^^^^^^^

A new flag - `-static-libclosure` was introduced to support statically linking
the runtime for the Blocks extension on Windows. This flag currently only
changes the code generation, and even then, only on Windows. This does not
impact the linker behaviour like the other `-static-*` flags.

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^

Improvements
^^^^^^^^^^^^

Moved checkers
^^^^^^^^^^^^^^

- After lots of improvements, the checker ``alpha.security.ArrayBoundV2`` is
  renamed to ``security.ArrayBound``. As this checker is stable now, the old
  checker ``alpha.security.ArrayBound`` (which was searching for the same kind
  of bugs with an different, simpler and less accurate algorithm) is removed.

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------

OpenMP Support
--------------
- Added support 'no_openmp_constructs' assumption clause.

Improvements
^^^^^^^^^^^^

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us on the `Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_.
