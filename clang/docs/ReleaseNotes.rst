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

AST Dumping Potentially Breaking Changes
----------------------------------------

Clang Frontend Potentially Breaking Changes
-------------------------------------------

Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------
- Parts of the interface returning string results will now return
  the empty string `""` when no result is available, instead of `None`.
- Calling a property on the `CompletionChunk` or `CompletionString` class
  statically now leads to an error, instead of returning a `CachedProperty` object
  that is used internally. Properties are only available on instances.

What's New in Clang |release|?
==============================
Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

C++ Language Changes
--------------------

C++17 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++14 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Removed the restriction to literal types in constexpr functions in C++23 mode.

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Compiler flags ``-std=c++2c`` and ``-std=gnu++2c`` have been added for experimental C++2c implementation work.
- Implemented `P2738R1: constexpr cast from void* <https://wg21.link/P2738R1>`_.
- Partially implemented `P2361R6: constexpr cast from void* <https://wg21.link/P2361R6>`_.
  The changes to attributes declarations are not part of this release.
- Implemented `P2741R3: user-generated static_assert messages  <https://wg21.link/P2741R3>`_.

- Add ``__builtin_is_virtual_base_of`` intrinsic, which supports
  `P2985R0 A type trait for detecting virtual base classes <https://wg21.link/p2985r0>`_

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C Language Changes
------------------

C2y Feature Support
^^^^^^^^^^^^^^^^^^^

C23 Feature Support
^^^^^^^^^^^^^^^^^^^

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

- Clang now disallows more than one ``__attribute__((ownership_returns(class, idx)))`` with
  different class names attached to one function.

- Introduced a new format attribute ``__attribute__((format(syslog, 1, 2)))`` from OpenBSD.

- The ``hybrid_patchable`` attribute is now supported on ARM64EC targets. It can be used to specify
  that a function requires an additional x86-64 thunk, which may be patched at runtime.

Improvements to Clang's diagnostics
-----------------------------------

- Some template related diagnostics have been improved.

  .. code-block:: c++

     void foo() { template <typename> int i; } // error: templates can only be declared in namespace or class scope

     struct S {
      template <typename> int i; // error: non-static data member 'i' cannot be declared as a template
     };

- Clang now has improved diagnostics for functions with explicit 'this' parameters. Fixes #GH97878

- Clang now diagnoses dangling references to fields of temporary objects. Fixes #GH81589.

- Clang now diagnoses undefined behavior in constant expressions more consistently. This includes invalid shifts, and signed overflow in arithmetic.

- -Wdangling-assignment-gsl is enabled by default.

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------

- Fixed the definition of ``ATOMIC_FLAG_INIT`` in ``<stdatomic.h>`` so it can
  be used in C++.

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a crash when an expression with a dependent ``__typeof__`` type is used as the operand of a unary operator. (#GH97646)
- Fixed a failed assertion when checking invalid delete operator declaration. (#GH96191)
- Fix a crash when checking destructor reference with an invalid initializer. (#GH97230)
- Clang now correctly parses potentially declarative nested-name-specifiers in pointer-to-member declarators.
- Fix a crash when checking the initialzier of an object that was initialized
  with a string literal. (#GH82167)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a crash in C due to incorrect lookup that members in nested anonymous struct/union
  can be found as ordinary identifiers in struct/union definition. (#GH31295)

- Fixed a crash caused by long chains of ``sizeof`` and other similar operators
  that can be followed by a non-parenthesized expression. (#GH45061)

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

X86 Support
^^^^^^^^^^^

- The MMX vector intrinsic functions from ``*mmintrin.h`` which
  operate on `__m64` vectors, such as ``_mm_add_pi8``, have been
  reimplemented to use the SSE2 instruction-set and XMM registers
  unconditionally. These intrinsics are therefore *no longer
  supported* if MMX is enabled without SSE2 -- either from targeting
  CPUs from the Pentium-MMX through the Pentium 3, or explicitly via
  passing arguments such as ``-mmmx -mno-sse2``. MMX assembly code
  remains supported without requiring SSE2, including inside
  inline-assembly.

- The compiler builtins such as ``__builtin_ia32_paddb`` which
  formerly implemented the above MMX intrinsic functions have been
  removed. Any uses of these removed functions should migrate to the
  functions defined by the ``*mmintrin.h`` headers. A mapping can be
  found in the file ``clang/www/builtins.py``.

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

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

- Fixed an issue with the `hasName` and `hasAnyName` matcher when matching
  inline namespaces with an enclosing namespace of the same name.

clang-format
------------

libclang
--------

Static Analyzer
---------------

New features
^^^^^^^^^^^^

- MallocChecker now checks for ``ownership_returns(class, idx)`` and ``ownership_takes(class, idx)``
  attributes with class names different from "malloc". Clang static analyzer now reports an error
  if class of allocation and deallocation function mismatches.
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#unix-mismatcheddeallocator-c-c>`__.

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^

Improvements
^^^^^^^^^^^^

- Improved the handling of the ``ownership_returns`` attribute. Now, Clang reports an
  error if the attribute is attached to a function that returns a non-pointer value.
  Fixes (#GH99501)

Moved checkers
^^^^^^^^^^^^^^

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------

OpenMP Support
--------------

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
