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

- For the ARM target, C-language intrinsics are now provided for the full Arm
  v8.1-M MVE instruction set. ``<arm_mve.h>`` supports the complete API defined
  in the Arm C Language Extensions.

- For the ARM target, C-language intrinsics ``<arm_cde.h>`` for the CDE
  instruction set are now provided.

- clang adds support for a set of  extended integer types (``_ExtInt(N)``) that
  permit non-power of 2 integers, exposing the LLVM integer types. Since a major
  motivating use case for these types is to limit 'bit' usage, these types don't
  automatically promote to 'int' when operations are done between two
  ``ExtInt(N)`` types, instead math occurs at the size of the largest
  ``ExtInt(N)`` type.

- Users of UBSan, PGO, and coverage on Windows will now need to add clang's
  library resource directory to their library search path. These features all
  use runtime libraries, and Clang provides these libraries in its resource
  directory. For example, if LLVM is installed in ``C:\Program Files\LLVM``,
  then the profile runtime library will appear at
  ``C:\Program Files\LLVM\lib\clang\11.0.0\lib\windows\clang_rt.profile-x86_64.lib``.
  To ensure that the linker can find the appropriate library, users should pass
  ``/LIBPATH:C:\Program Files\LLVM\lib\clang\11.0.0\lib\windows`` to the
  linker. If the user links the program with the ``clang`` or ``clang-cl``
  drivers, the driver will pass this flag for them.

New Compiler Flags
------------------

- -fstack-clash-protection will provide a protection against the stack clash
  attack for x86 architecture through automatic probing of each page of
  allocated stack.

- -ffp-exception-behavior={ignore,maytrap,strict} allows the user to specify
  the floating-point exception behavior. The default setting is ``ignore``.

- -ffp-model={precise,strict,fast} provides the user an umbrella option to
  simplify access to the many single purpose floating point options. The default
  setting is ``precise``.

Deprecated Compiler Flags
-------------------------

The following options are deprecated and ignored. They will be removed in
future versions of Clang.

- ...

Modified Compiler Flags
-----------------------

- -fno-common has been enabled as the default for all targets.  Therefore, C
  code that uses tentative definitions as definitions of a variable in multiple
  translation units will trigger multiple-definition linker errors. Generally,
  this occurs when the use of the ``extern`` keyword is neglected in the
  declaration of a variable in a header file. In some cases, no specific
  translation unit provides a definition of the variable. The previous
  behavior can be restored by specifying ``-fcommon``.
- -Wasm-ignored-qualifier (ex. `asm const ("")`) has been removed and replaced
  with an error (this matches a recent change in GCC-9).
- -Wasm-file-asm-volatile (ex. `asm volatile ("")` at global scope) has been
  removed and replaced with an error (this matches GCC's behavior).
- Duplicate qualifiers on asm statements (ex. `asm volatile volatile ("")`) no
  longer produces a warning via -Wduplicate-decl-specifier, but now an error
  (this matches GCC's behavior).
- The deprecated argument ``-f[no-]sanitize-recover`` has changed to mean
  ``-f[no-]sanitize-recover=all`` instead of
  ``-f[no-]sanitize-recover=undefined,integer`` and is no longer deprecated.
- The argument to ``-f[no-]sanitize-trap=...`` is now optional and defaults to
  ``all``.
- ``-fno-char8_t`` now disables the ``char8_t`` keyword, not just the use of
  ``char8_t`` as the character type of ``u8`` literals. This restores the
  Clang 8 behavior that regressed in Clang 9 and 10.
- -print-targets has been added to print the registered targets.

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- Attributes can now be specified by clang plugins. See the
  `Clang Plugins <ClangPlugins.html#defining-attributes>`_ documentation for
  details.

Windows Support
---------------

C Language Changes in Clang
---------------------------

- The default C language standard used when `-std=` is not specified has been
  upgraded from gnu11 to gnu17.

- Clang now supports the GNU C extension `asm inline`; it won't do anything
  *yet*, but it will be parsed.

- ...

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
  in Clang 11. In addition, cases that previous versions of Clang did not
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

- clang-tidy and clang-include-fixer are no longer compiled into libclang by
  default. You can set ``LIBCLANG_INCLUDE_CLANG_TOOLS_EXTRA=ON`` to undo that,
  but it's expected that that setting will go away eventually. If this is
  something you need, please reach out to the mailing list to discuss possible
  ways forward.

AST Matchers
------------

- ...

clang-format
------------

- Option ``IndentCaseBlocks`` has been added to support treating the block
  following a switch case label as a scope block which gets indented itself.
  It helps avoid having the closing bracket align with the switch statement's
  closing bracket (when ``IndentCaseLabels`` is ``false``).

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

- Option ``ObjCBreakBeforeNestedBlockParam`` has been added to optionally apply
  linebreaks for function arguments declarations before nested blocks.

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
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us via the `mailing
list <https://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
