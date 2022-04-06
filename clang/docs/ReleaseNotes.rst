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
     `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release |release|. Here we
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

What's New in Clang |release|?
==============================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Clang now supports the ``-fzero-call-used-regs`` feature for x86. The purpose
  of this feature is to limit Return-Oriented Programming (ROP) exploits and
  information leakage. It works by zeroing out a selected class of registers
  before function return --- e.g., all GPRs that are used within the function.
  There is an analogous ``zero_call_used_regs`` attribute to allow for finer
  control of this feature.

Bug Fixes
------------------
- ``CXXNewExpr::getArraySize()`` previously returned a ``llvm::Optional``
  wrapping a ``nullptr`` when the ``CXXNewExpr`` did not have an array
  size expression. This was fixed and ``::getArraySize()`` will now always
  either return ``None`` or a ``llvm::Optional`` wrapping a valid ``Expr*``.
  This fixes `Issue 53742 <https://github.com/llvm/llvm-project/issues/53742>`_.
- We now ignore full expressions when traversing cast subexpressions. This
  fixes `Issue 53044 <https://github.com/llvm/llvm-project/issues/53044>`_.
- Allow `-Wno-gnu` to silence GNU extension diagnostics for pointer arithmetic
  diagnostics. Fixes `Issue 54444 <https://github.com/llvm/llvm-project/issues/54444>`_.
- Placeholder constraints, as in `Concept auto x = f();`, were not checked when modifiers
  like ``auto&`` or ``auto**`` were added. These constraints are now checked.
  This fixes  `Issue 53911 <https://github.com/llvm/llvm-project/issues/53911>`_
  and  `Issue 54443 <https://github.com/llvm/llvm-project/issues/54443>`_.
- Previously invalid member variables with template parameters would crash clang.
  Now fixed by setting identifiers for them.
  This fixes `Issue 28475 (PR28101) <https://github.com/llvm/llvm-project/issues/28475>`_.
- Now allow the `restrict` and `_Atomic` qualifiers to be used in conjunction
  with `__auto_type` to match the behavior in GCC. This fixes
  `Issue 53652 <https://github.com/llvm/llvm-project/issues/53652>`_.
- No longer crash when specifying a variably-modified parameter type in a
  function with the ``naked`` attribute. This fixes
  `Issue 50541 <https://github.com/llvm/llvm-project/issues/50541>`_.
- Allow multiple ``#pragma weak`` directives to name the same undeclared (if an
  alias, target) identifier instead of only processing one such ``#pragma weak``
  per identifier.
  Fixes `Issue 28985 <https://github.com/llvm/llvm-project/issues/28985>`_.
- Assignment expressions in C11 and later mode now properly strip the _Atomic
  qualifier when determining the type of the assignment expression. Fixes
  `Issue 48742 <https://github.com/llvm/llvm-project/issues/48742>`_.
- Improved the diagnostic when accessing a member of an atomic structure or
  union object in C; was previously an unhelpful error, but now issues a
  `-Watomic-access` warning which defaults to an error. Fixes
  `Issue 54563 <https://github.com/llvm/llvm-project/issues/54563>`_.
- Unevaluated lambdas in dependant contexts no longer result in clang crashing.
  This fixes Issues `50376 <https://github.com/llvm/llvm-project/issues/50376>`_,
  `51414 <https://github.com/llvm/llvm-project/issues/51414>`_,
  `51416 <https://github.com/llvm/llvm-project/issues/51416>`_,
  and `51641 <https://github.com/llvm/llvm-project/issues/51641>`_.
- The builtin function __builtin_dump_struct would crash clang when the target 
  struct contains a bitfield. It now correctly handles bitfields.
  This fixes Issue `Issue 54462 <https://github.com/llvm/llvm-project/issues/54462>`_.
- Statement expressions are now disabled in default arguments in general.
  This fixes Issue `Issue 53488 <https://github.com/llvm/llvm-project/issues/53488>`_.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``-Wliteral-range`` will warn on floating-point equality comparisons with
  constants that are not representable in a casted value. For example,
  ``(float) f == 0.1`` is always false.
- ``-Winline-namespace-reopened-noninline`` now takes into account that the
  ``inline`` keyword must appear on the original but not necessarily all
  extension definitions of an inline namespace and therefore points its note
  at the original definition. This fixes `Issue 50794 (PR51452)
  <https://github.com/llvm/llvm-project/issues/50794>`_.
- ``-Wunused-but-set-variable`` now also warns if the variable is only used
  by unary operators.
- ``-Wunused-variable`` no longer warn for references extending the lifetime
  of temporaries with side effects. This fixes `Issue 54489
  <https://github.com/llvm/llvm-project/issues/54489>`_.

Non-comprehensive list of changes in this release
-------------------------------------------------
- Improve __builtin_dump_struct:
  - Support bitfields in struct and union.
  - Improve the dump format, dump both bitwidth(if its a bitfield) and field value.
  - Remove anonymous tag locations.
  - Beautify dump format, add indent for nested struct and struct members.

New Compiler Flags
------------------

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
-------------------------

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- Added support for parameter pack expansion in `clang::annotate`.

- The ``overloadable`` attribute can now be written in all of the syntactic
  locations a declaration attribute may appear.
  This fixes `Issue 53805 <https://github.com/llvm/llvm-project/issues/53805>`_.

- Improved namespace attributes handling:

  - Handle GNU attributes before a namespace identifier and subsequent
    attributes of different kinds.
  - Emit error on GNU attributes for a nested namespace definition.

- Statement attributes ``[[clang::noinline]]`` and  ``[[clang::always_inline]]``
  can be used to control inlining decisions at callsites.

- ``#pragma clang attribute push`` now supports multiple attributes within a single directive.

- The ``__declspec(naked)`` attribute can no longer be written on a member
  function in Microsoft compatibility mode, matching the behavior of cl.exe.

Windows Support
---------------

- Add support for MSVC-compatible ``/JMC``/``/JMC-`` flag in clang-cl (supports
  X86/X64/ARM/ARM64). ``/JMC`` could only be used when ``/Zi`` or ``/Z7`` is
  turned on. With this addition, clang-cl can be used in Visual Studio for the
  JustMyCode feature. Note, you may need to manually add ``/JMC`` as additional
  compile options in the Visual Studio since it currently assumes clang-cl does not support ``/JMC``.

C Language Changes in Clang
---------------------------

C2x Feature Support
-------------------

- Implemented `WG14 N2674 The noreturn attribute <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2764.pdf>`_.
- Implemented `WG14 N2935 Make false and true first-class language features <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2935.pdf>`_.
- Implemented `WG14 N2763 Adding a fundamental type for N-bit integers <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2763.pdf>`_.
- Implemented `WG14 N2775 Literal suffixes for bit-precise integers <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2775.pdf>`_.
- Implemented the `*_WIDTH` macros to complete support for
  `WG14 N2412 Two's complement sign representation for C2x <https://www9.open-std.org/jtc1/sc22/wg14/www/docs/n2412.pdf>`_.

C++ Language Changes in Clang
-----------------------------

- ...

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Diagnose consteval and constexpr issues that happen at namespace scope. This
  partially addresses `Issue 51593 <https://github.com/llvm/llvm-project/issues/51593>`_.
- No longer attempt to evaluate a consteval UDL function call at runtime when
  it is called through a template instantiation. This fixes
  `Issue 54578 <https://github.com/llvm/llvm-project/issues/54578>`_.

- Implemented `__builtin_source_location()` which enables library support for std::source_location.

C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P2128R6: Multidimensional subscript operator <https://wg21.link/P2128R6>`_.
- Implemented `P0849R8: auto(x): decay-copy in the language <https://wg21.link/P0849R8>`_.
- Implemented `P2242R3: Non-literal variables (and labels and gotos) in constexpr functions	<https://wg21.link/P2242R3>`_.

CUDA Language Changes in Clang
------------------------------

Objective-C Language Changes in Clang
-------------------------------------

OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------

OpenMP Support in Clang
-----------------------

- ``clang-nvlink-wrapper`` tool introduced to support linking of cubin files
  archived in an archive. See :doc:`ClangNvlinkWrapper`.
- ``clang-linker-wrapper`` tool introduced to support linking using a new OpenMP
  target offloading method. See :doc:`ClangLinkerWrapper`.
- Support for a new driver for OpenMP target offloading has been added as an
  opt-in feature. The new driver can be selected using ``-fopenmp-new-driver``
  with clang. Device-side LTO can also be enabled using the new driver by
  passing ``-foffload-lto=`` as well. The new driver supports the following
  features:
  - Linking AMDGPU and NVPTX offloading targets.
  - Static linking using archive files.
  - Device-side LTO.

CUDA Support in Clang
---------------------

- ...

X86 Support in Clang
--------------------

DWARF Support in Clang
----------------------

Arm and AArch64 Support in Clang
--------------------------------

- When using ``-mbranch-protection=bti`` with AArch64, calls to setjmp will
  now be followed by a BTI instruction. This is done to be compatible with
  setjmp implementations that return with a br instead of a ret. You can
  disable this behaviour using the ``-mno-bti-at-return-twice`` option.

Floating Point Support in Clang
-------------------------------

Internal API Changes
--------------------

- Added a new attribute flag `AcceptsExprPack` that when set allows expression
  pack expansions in the parsed arguments of the corresponding attribute.
  Additionally it introduces delaying of attribute arguments, adding common
  handling for creating attributes that cannot be fully initialized prior to
  template instantiation.

Build System Changes
--------------------

AST Matchers
------------

- Expanded ``isInline`` narrowing matcher to support c++17 inline variables.

clang-format
------------

- **Important change**: Renamed ``IndentRequires`` to ``IndentRequiresClause``
  and changed the default for all styles from ``false`` to ``true``.

- Reworked and improved handling of concepts and requires. Added the
  ``RequiresClausePosition`` option as part of that.

- Changed ``BreakBeforeConceptDeclarations`` from ``Boolean`` to an enum.

- Option ``InsertBraces`` has been added to insert optional braces after control
  statements.

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
