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
- Indirect edges of asm goto statements under certain circumstances may now be
  split. In previous releases of clang, that means for the following code the
  two inputs may have compared equal in the inline assembly.  This is no longer
  guaranteed (and necessary to support outputs along indirect edges, which is
  now supported as of this release). This change is more consistent with the
  behavior of GCC.

  .. code-block:: c

    foo: asm goto ("# %0 %1"::"i"(&&foo)::foo);

C++ Specific Potentially Breaking Changes
-----------------------------------------
- Clang won't search for coroutine_traits in std::experimental namespace any more.
  Clang will only search for std::coroutine_traits for coroutines then.

ABI Changes in This Version
---------------------------
- ``__is_trivial`` has changed for a small category of classes with constrained default constructors (`#60697 <https://github.com/llvm/llvm-project/issues/60697>`_).
  *FIXME: Remove this note if we've backported this change to the Clang 16 branch.*

What's New in Clang |release|?
==============================
Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

C++ Language Changes
--------------------
- Improved ``-O0`` code generation for calls to ``std::forward_like``. Similarly to
  ``std::move, std::forward`` et al. it is now treated as a compiler builtin and implemented
  directly rather than instantiating the definition from the standard library.
- Implemented `CWG2518 <https://wg21.link/CWG2518>`_ which allows ``static_assert(false)``
  to not be ill-formed when its condition is evaluated in the context of a template definition.

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P2036R3: Change scope of lambda trailing-return-type <https://wg21.link/P2036R3>`_
  and `P2579R0 Mitigation strategies for P2036 <https://wg21.link/P2579R0>`_.
  These proposals modify how variables captured in lambdas can appear in trailing return type
  expressions and how their types are deduced therein, in all C++ language versions.

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C Language Changes
------------------
- Support for outputs from asm goto statements along indirect edges has been
  added. This fixes
  `Issue 53562 <https://github.com/llvm/llvm-project/issues/53562>`_.

C2x Feature Support
^^^^^^^^^^^^^^^^^^^
- Implemented the ``unreachable`` macro in freestanding ``<stddef.h>`` for
  `WG14 N2826 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2826.pdf>`_

- Removed the ``ATOMIC_VAR_INIT`` macro in C2x and later standards modes, which
  implements `WG14 N2886 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2886.htm>`_

- Implemented `WG14 N2934 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2934.pdf>`_
  which introduces the ``bool``, ``static_assert``, ``alignas``, ``alignof``,
  and ``thread_local`` keywords in C2x.

Non-comprehensive list of changes in this release
-------------------------------------------------
- Clang now saves the address of ABI-indirect function parameters on the stack,
  improving the debug information available in programs compiled without
  optimizations.
- Clang now supports ``__builtin_nondeterministic_value`` that returns a
  nondeterministic value of the same type as the provided argument.

New Compiler Flags
------------------

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
-------------------------
- The deprecated flag `-fmodules-ts` is removed. Please use ``-std=c++20``
  or higher to use standard C++ modules instead.
- The deprecated flag `-fcoroutines-ts` is removed. Please use ``-std=c++20``
  or higher to use standard C++ coroutines instead.

Attribute Changes in Clang
--------------------------
- Introduced a new function attribute ``__attribute__((unsafe_buffer_usage))``
  to be worn by functions containing buffer operations that could cause out of
  bounds memory accesses. It emits warnings at call sites to such functions when
  the flag ``-Wunsafe-buffer-usage`` is enabled.
- ``__declspec`` attributes can now be used together with the using keyword. Before
  the attributes on ``__declspec`` was ignored, while now it will be forwarded to the
  point where the alias is used.
- Introduced a new ``USR`` (unified symbol resolution) clause inside of the
  existing ``__attribute__((external_source_symbol))`` attribute. Clang's indexer
  uses the optional USR value when indexing Clang's AST. This value is expected
  to be generated by an external compiler when generating C++ bindings during
  the compilation of the foreign language sources (e.g. Swift).

Improvements to Clang's diagnostics
-----------------------------------
- We now generate a diagnostic for signed integer overflow due to unary minus
  in a non-constant expression context.
  (`#31643 <https://github.com/llvm/llvm-project/issues/31643>`)
- Clang now warns by default for C++20 and later about deprecated capture of
  ``this`` with a capture default of ``=``. This warning can be disabled with
  ``-Wno-deprecated-this-capture``.
- Clang had failed to emit some ``-Wundefined-internal`` for members of a local
  class if that class was first introduced with a forward declaration.
- Diagnostic notes and fix-its are now generated for ``ifunc``/``alias`` attributes
  which point to functions whose names are mangled.
- Diagnostics relating to macros on the command line of a preprocessed assembly
  file are now reported as coming from the file ``<command line>`` instead of
  ``<built-in>``.

Bug Fixes in This Version
-------------------------

- Fix crash when diagnosing incorrect usage of ``_Nullable`` involving alias
  templates.
  (`#60344 <https://github.com/llvm/llvm-project/issues/60344>`_)
- Fix confusing warning message when ``/clang:-x`` is passed in ``clang-cl``
  driver mode and emit an error which suggests using ``/TC`` or ``/TP``
  ``clang-cl`` options instead.
  (`#59307 <https://github.com/llvm/llvm-project/issues/59307>`_)
- Fix assert that fails when the expression causing the this pointer to be
  captured by a block is part of a constexpr if statement's branch and
  instantiation of the enclosing method causes the branch to be discarded.
- Fix __VA_OPT__ implementation so that it treats the concatenation of a
  non-placemaker token and placemaker token as a non-placemaker token.
  (`#60268 <https://github.com/llvm/llvm-project/issues/60268>`_)
- Fix crash when taking the address of a consteval lambda call operator.
  (`#57682 <https://github.com/llvm/llvm-project/issues/57682>`_)
- Clang now support export declarations in the language linkage.
  (`#60405 <https://github.com/llvm/llvm-project/issues/60405>`_)
- Fix aggregate initialization inside lambda constexpr.
  (`#60936 <https://github.com/llvm/llvm-project/issues/60936>`_)

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Fix crash on invalid code when looking up a destructor in a templated class
  inside a namespace.
  (`#59446 <https://github.com/llvm/llvm-project/issues/59446>`_)
- Fix crash when evaluating consteval constructor of derived class whose base
  has more than one field.
  (`#60166 <https://github.com/llvm/llvm-project/issues/60166>`_)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Target Specific Changes
-----------------------

X86 Support
^^^^^^^^^^^

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

- The hard-float ABI is now available in Armv8.1-M configurations that
  have integer MVE instructions (and therefore have FP registers) but
  no scalar or vector floating point computation. Previously, trying
  to select the hard-float ABI on such a target (via
  ``-mfloat-abi=hard`` or a triple ending in ``hf``) would silently
  use the soft-float ABI instead.

- Clang builtin ``__arithmetic_fence`` and the command line option ``-fprotect-parens``
  are now enabled for AArch64.

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^
- Added ``-mrvv-vector-bits=`` option to give an upper and lower bound on vector
  length. Valid values are powers of 2 between 64 and 65536. A value of 32
  should eventually be supported. We also accept "zvl" to use the Zvl*b
  extension from ``-march`` or ``-mcpu`` to the be the upper and lower bound.

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
- The definition of ``USHRT_MAX`` in the freestanding ``<limits.h>`` no longer
  overflows on AVR (where ``sizeof(int) == sizeof(unsigned short)``).  The type
  of ``USHRT_MAX`` is now ``unsigned int`` instead of ``int``, as required by
  the C standard.

DWARF Support in Clang
----------------------

Floating Point Support in Clang
-------------------------------
- Add ``__builtin_elementwise_log`` builtin for floating point types only.
- Add ``__builtin_elementwise_log10`` builtin for floating point types only.
- Add ``__builtin_elementwise_log2`` builtin for floating point types only.
- Add ``__builtin_elementwise_exp`` builtin for floating point types only.
- Add ``__builtin_elementwise_exp2`` builtin for floating point types only.

AST Matchers
------------

- Add ``coroutineBodyStmt`` matcher.

- The ``hasBody`` matcher now matches coroutine body nodes in
  ``CoroutineBodyStmts``.

clang-format
------------

- Add ``NextLineOnly`` style to option ``PackConstructorInitializers``.
  Compared to ``NextLine`` style, ``NextLineOnly`` style will not try to
  put the initializers on the current line first, instead, it will try to
  put the initializers on the next line only.

libclang
--------

- Introduced the new function ``clang_CXXMethod_isExplicit``,
  which identifies whether a constructor or conversion function cursor
  was marked with the explicit identifier.

- Introduced the new ``CXIndex`` constructor function
  ``clang_createIndexWithOptions``, which allows overriding precompiled preamble
  storage path.

- Deprecated two functions ``clang_CXIndex_setGlobalOptions`` and
  ``clang_CXIndex_setInvocationEmissionPathOption`` in favor of the new
  function ``clang_createIndexWithOptions`` in order to improve thread safety.

Static Analyzer
---------------
- Fix incorrect alignment attribute on the this parameter of certain
  non-complete destructors when using the Microsoft ABI.
  `Issue 60465 <https://github.com/llvm/llvm-project/issues/60465>`_.

.. _release-notes-sanitizers:

Sanitizers
----------


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
