========================================
Clang 12.0.0 (In-Progress) Release Notes
========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 12 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 12.0.0. Here we
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

What's New in Clang 12.0.0?
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

- The builtin intrinsics ``__builtin_bitreverse8``, ``__builtin_bitreverse16``,
  ``__builtin_bitreverse32`` and ``__builtin_bitreverse64`` may now be used
  within constant expressions.

- The builtin intrinsics ``__builtin_rotateleft8``, ``__builtin_rotateleft16``,
  ``__builtin_rotateleft32`` and ``__builtin_rotateleft64`` may now be used
  within constant expressions.

- The builtin intrinsics ``__builtin_rotateright8``, ``__builtin_rotateright16``,
  ``__builtin_rotateright32`` and ``__builtin_rotateright64`` may now be used
  within constant expressions.

New Compiler Flags
------------------

- ...

- -fpch-codegen and -fpch-debuginfo generate shared code and/or debuginfo
  for contents of a precompiled header in a separate object file. This object
  file needs to be linked in, but its contents do not need to be generated
  for other objects using the precompiled header. This should usually save
  compile time. If not using clang-cl, the separate object file needs to
  be created explicitly from the precompiled header.
  Example of use:

  .. code-block:: console

    $ clang++ -x c++-header header.h -o header.pch -fpch-codegen -fpch-debuginfo
    $ clang++ -c header.pch -o shared.o
    $ clang++ -c source.cpp -o source.o -include-pch header.pch
    $ clang++ -o binary source.o shared.o

  - Using -fpch-instantiate-templates when generating the precompiled header
    usually increases the amount of code/debuginfo that can be shared.
  - In some cases, especially when building with optimizations enabled, using
    -fpch-codegen may generate so much code in the shared object that compiling
    it may be a net loss in build time.
  - Since headers may bring in private symbols of other libraries, it may be
    sometimes necessary to discard unused symbols (such as by adding
    -Wl,--gc-sections on ELF platforms to the linking command, and possibly
    adding -fdata-sections -ffunction-sections to the command generating
    the shared object).

Deprecated Compiler Flags
-------------------------

The following options are deprecated and ignored. They will be removed in
future versions of Clang.

- ...

Modified Compiler Flags
-----------------------

- On ELF, ``-gz`` now defaults to ``-gz=zlib`` with the integrated assembler.
  It produces ``SHF_COMPRESSED`` style compression of debug information. GNU
  binutils 2.26 or newer, or lld is required to link produced object files. Use
  ``-gz=zlib-gnu`` to get the old behavior.
- Now that `this` pointers are tagged with `nonnull` and `dereferenceable(N)`,
  `-fno-delete-null-pointer-checks` has gained the power to remove the
  `nonnull` attribute on `this` for configurations that need it to be nullable.
- ``-gsplit-dwarf`` no longer implies ``-g2``.
- ``-fasynchronous-unwind-tables`` is now the default on Linux AArch64/PowerPC.
  This behavior matches newer GCC.
  (`D91760 <https://reviews.llvm.org/D91760>`_)
  (`D92054 <https://reviews.llvm.org/D92054>`_)

Removed Compiler Flags
-------------------------

The following options no longer exist.

- clang-cl's ``/Zd`` flag no longer exist. But ``-gline-tables-only`` still
  exists and does the same thing.

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- Added support for the C++20 likelihood attributes ``[[likely]]`` and
  ``[[unlikely]]``. As an extension they can be used in C++11 and newer.
  This extension is enabled by default.

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

- The x86 intrinsics ``_mm_popcnt_u32``, ``_mm_popcnt_u64``, ``_popcnt32``,
  ``_popcnt64``, ``__popcntd`` and ``__popcntq``  may now be used within
  constant expressions.

- The x86 intrinsics ``_bit_scan_forward``, ``__bsfd`` and ``__bsfq`` may now
  be used within constant expressions.

- The x86 intrinsics ``_bit_scan_reverse``, ``__bsrd`` and ``__bsrq`` may now
  be used within constant expressions.

- The x86 intrinsics ``__bswap``, ``__bswapd``, ``__bswap64`` and ``__bswapq``
  may now be used within constant expressions.

- The x86 intrinsics ``_castf32_u32``, ``_castf64_u64``, ``_castu32_f32`` and
  ``_castu64_f64`` may now be used within constant expressions.

- The x86 intrinsics ``__rolb``, ``__rolw``, ``__rold``, ``__rolq`, ``_rotl``,
  ``_rotwl`` and ``_lrotl`` may now be used within constant expressions.

- The x86 intrinsics ``__rorb``, ``__rorw``, ``__rord``, ``__rorq`, ``_rotr``,
  ``_rotwr`` and ``_lrotr`` may now be used within constant expressions.

- Support for ``-march=alderlake``, ``-march=sapphirerapids`` and
  ``-march=znver3`` was added.

- Support for ``-march=x86-64-v[234]`` has been added.
  See :doc:`UsersManual` for details about these micro-architecture levels.

- The -mtune command line option is no longer ignored for X86. This can be used
  to request microarchitectural optimizations independent on -march. -march=<cpu>
  implies -mtune=<cpu>. -mtune=generic is the default with no -march or -mtune
  specified.

- Support for ``HRESET`` instructions has been added.

- Support for ``UINTR`` instructions has been added.

- Support for ``AVXVNNI`` instructions has been added.

Internal API Changes
--------------------

These are major API changes that have happened since the 11.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

- ...

Build System Changes
--------------------

These are major changes to the build system that have happened since the 11.0.0
release of Clang. Users of the build system should adjust accordingly.

- ...

AST Matchers
------------

- The behavior of TK_IgnoreUnlessSpelledInSource with the traverse() matcher
  has been changed to no longer match on template instantiations or on
  implicit nodes which are not spelled in the source.

- The TK_IgnoreImplicitCastsAndParentheses traversal kind was removed. It
  is recommended to use TK_IgnoreUnlessSpelledInSource instead.

- The behavior of the forEach() matcher was changed to not internally ignore
  implicit and parenthesis nodes.

clang-format
------------

- Option ``BitFieldColonSpacing`` has been added that decides how
  space should be added around identifier, colon and bit-width in
  bitfield definitions.

  .. code-block:: c++

    // Both (default)
    struct F {
      unsigned dscp : 6;
      unsigned ecn  : 2; // AlignConsecutiveBitFields=true
    };
    // None
    struct F {
      unsigned dscp:6;
      unsigned ecn :2;
    };
    // Before
    struct F {
      unsigned dscp :6;
      unsigned ecn  :2;
    };
    // After
    struct F {
      unsigned dscp: 6;
      unsigned ecn : 2;
    };


- Experimental Support in clang-format for concepts has been improved, to
  aid this the follow options have been added

- Option ``IndentRequires`` has been added to indent the ``requires`` keyword
  in templates.
- Option ``BreakBeforeConceptDeclarations`` has been added to aid the formatting of concepts.

- Option ``IndentPragmas`` has been added to allow #pragma to indented with the current scope level. This is especially useful when using #pragma to mark OpenMP sections of code.

- Option ``SpaceBeforeCaseColon`` has been added to add a space before the
  colon in a case or default statement.


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
