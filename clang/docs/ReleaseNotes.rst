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
These changes are ones which we think may surprise users when upgrading to
Clang |release| because of the opportunity they pose for disruption to existing
code bases.


C/C++ Language Potentially Breaking Changes
-------------------------------------------

C++ Specific Potentially Breaking Changes
-----------------------------------------

ABI Changes in This Version
---------------------------

What's New in Clang |release|?
==============================
Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

C++ Language Changes
--------------------

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P2169R4: A nice placeholder with no name <https://wg21.link/P2169R4>`_. This allows using ``_``
  as a variable name multiple times in the same scope and is supported in all C++ language modes as an extension.
  An extension warning is produced when multiple variables are introduced by ``_`` in the same scope.
  Unused warnings are no longer produced for variables named ``_``.
  Currently, inspecting placeholders variables in a debugger when more than one are declared in the same scope
  is not supported.

  .. code-block:: cpp
    struct S {
      int _, _; // Was invalid, now OK
    };
    void func() {
      int _, _; // Was invalid, now OK
    }
    void other() {
      int _; // Previously diagnosed under -Wunused, no longer diagnosed
    }



Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C Language Changes
------------------
- ``structs``, ``unions``, and ``arrays`` that are const may now be used as
  constant expressions.  This change is more consistent with the behavior of
  GCC.

C2x Feature Support
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

Improvements to Clang's diagnostics
-----------------------------------
- Clang constexpr evaluator now prints template arguments when displaying
  template-specialization function calls.

Bug Fixes in This Version
-------------------------
- Fixed an issue where a class template specialization whose declaration is
  instantiated in one module and whose definition is instantiated in another
  module may end up with members associated with the wrong declaration of the
  class, which can result in miscompiles in some cases.
- Fix crash on use of a variadic overloaded operator.
  (`#42535 <https://github.com/llvm/llvm-project/issues/42535>_`)

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Clang limits the size of arrays it will try to evaluate at compile time
  to avoid memory exhaustion.
  This limit can be modified by `-fconstexpr-steps`.
  (`#63562 <https://github.com/llvm/llvm-project/issues/63562>`_)

- Fix a crash caused by some named unicode escape sequences designating
  a Unicode character whose name contains a ``-``.
  (`Fixes #64161 <https://github.com/llvm/llvm-project/issues/64161>_`)

- Fix cases where we ignore ambiguous name lookup when looking up memebers.
  (`#22413 <https://github.com/llvm/llvm-project/issues/22413>_`),
  (`#29942 <https://github.com/llvm/llvm-project/issues/29942>_`),
  (`#35574 <https://github.com/llvm/llvm-project/issues/35574>_`) and
  (`#27224 <https://github.com/llvm/llvm-project/issues/27224>_`).

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed an import failure of recursive friend class template.
  `Issue 64169 <https://github.com/llvm/llvm-project/issues/64169>`_
- Remove unnecessary RecordLayout computation when importing UnaryOperator. The
  computed RecordLayout is incorrect if fields are not completely imported and
  should not be cached.
  `Issue 64170 <https://github.com/llvm/llvm-project/issues/64170>`_

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed a crash when parsing top-level ObjC blocks that aren't properly
  terminated. Clang should now also recover better when an @end is missing
  between blocks.
  `Issue 64065 <https://github.com/llvm/llvm-project/issues/64065>`_

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

X86 Support
^^^^^^^^^^^

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^
- Unaligned memory accesses can be toggled by ``-m[no-]unaligned-access`` or the
  aliases ``-m[no-]strict-align``.

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA Support
^^^^^^^^^^^^

AIX Support
^^^^^^^^^^^

WebAssembly Support
^^^^^^^^^^^^^^^^^^^

AVR Support
^^^^^^^^^^^

DWARF Support in Clang
----------------------

Floating Point Support in Clang
-------------------------------
- Add ``__builtin_elementwise_log`` builtin for floating point types only.
- Add ``__builtin_elementwise_log10`` builtin for floating point types only.
- Add ``__builtin_elementwise_log2`` builtin for floating point types only.
- Add ``__builtin_elementwise_exp`` builtin for floating point types only.
- Add ``__builtin_elementwise_exp2`` builtin for floating point types only.
- Add ``__builtin_set_flt_rounds`` builtin for X86, x86_64, Arm and AArch64 only.
- Add ``__builtin_elementwise_pow`` builtin for floating point types only.
- Add ``__builtin_elementwise_bitreverse`` builtin for integer types only.

AST Matchers
------------

clang-format
------------

libclang
--------

Static Analyzer
---------------

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------

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
