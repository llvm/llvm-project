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

- Attributes now expect unevaluated strings in attributes parameters that are string literals.
  This is applied to both C++ standard attributes, and other attributes supported by Clang.
  This completes the implementation of `P2361R6 Unevaluated Strings <https://wg21.link/P2361R6>`_


Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Implemented `CWG1473 <https://wg21.link/CWG1473>`_ which allows spaces after ``operator""``.
  Clang used to err on the lack of space when the literal suffix identifier was invalid in
  all the language modes, which contradicted the deprecation of the whitespaces.
  Also turn on ``-Wdeprecated-literal-operator`` by default in all the language modes.

C Language Changes
------------------
- ``structs``, ``unions``, and ``arrays`` that are const may now be used as
  constant expressions.  This change is more consistent with the behavior of
  GCC.

C23 Feature Support
^^^^^^^^^^^^^^^^^^^
- Clang now accepts ``-std=c23`` and ``-std=gnu23`` as language standard modes,
  and the ``__STDC_VERSION__`` macro now expands to ``202311L`` instead of its
  previous placeholder value. Clang continues to accept ``-std=c2x`` and
  ``-std=gnu2x`` as aliases for C23 and GNU C23, respectively.

Non-comprehensive list of changes in this release
-------------------------------------------------

New Compiler Flags
------------------

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

* ``-Woverriding-t-option`` is renamed to ``-Woverriding-option``.

Removed Compiler Flags
-------------------------

Attribute Changes in Clang
--------------------------

Improvements to Clang's diagnostics
-----------------------------------
- Clang constexpr evaluator now prints template arguments when displaying
  template-specialization function calls.
- Clang contexpr evaluator now displays notes as well as an error when a constructor
  of a base class is not called in the constructor of its derived class.
- Clang no longer emits ``-Wmissing-variable-declarations`` for variables declared
  with the ``register`` storage class.
- Clang's ``-Wtautological-negation-compare`` flag now diagnoses logical
  tautologies like ``x && !x`` and ``!x || x`` in expressions. This also
  makes ``-Winfinite-recursion`` diagnose more cases.
  (`#56035: <https://github.com/llvm/llvm-project/issues/56035>`_).

Bug Fixes in This Version
-------------------------
- Fixed an issue where a class template specialization whose declaration is
  instantiated in one module and whose definition is instantiated in another
  module may end up with members associated with the wrong declaration of the
  class, which can result in miscompiles in some cases.
- Fix crash on use of a variadic overloaded operator.
  (`#42535 <https://github.com/llvm/llvm-project/issues/42535>_`)
- Fix a hang on valid C code passing a function type as an argument to
  ``typeof`` to form a function declaration.
  (`#64713 <https://github.com/llvm/llvm-project/issues/64713>_`)

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
  (Fixes `#64161 <https://github.com/llvm/llvm-project/issues/64161>`_)

- Fix cases where we ignore ambiguous name lookup when looking up members.
  (`#22413 <https://github.com/llvm/llvm-project/issues/22413>`_),
  (`#29942 <https://github.com/llvm/llvm-project/issues/29942>`_),
  (`#35574 <https://github.com/llvm/llvm-project/issues/35574>`_) and
  (`#27224 <https://github.com/llvm/llvm-project/issues/27224>`_).

- Clang emits an error on substitution failure within lambda body inside a
  requires-expression. This fixes:
  (`#64138 <https://github.com/llvm/llvm-project/issues/64138>`_).

- Update ``FunctionDeclBitfields.NumFunctionDeclBits``. This fixes:
  (`#64171 <https://github.com/llvm/llvm-project/issues/64171>`_).

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
- Fixed a crash when check array access on zero-length element.
  `Issue 64564 <https://github.com/llvm/llvm-project/issues/64564>`_

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^
- Use pass-by-reference (byref) in stead of pass-by-value (byval) for struct
  arguments in C ABI. Callee is responsible for allocating stack memory and
  copying the value of the struct if modified. Note that AMDGPU backend still
  supports byval for struct arguments.

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
- Add ``__builtin_elementwise_sqrt`` builtin for floating point types only.

AST Matchers
------------
- Add ``convertVectorExpr``.
- Add ``dependentSizedExtVectorType``.
- Add ``macroQualifiedType``.

clang-format
------------

libclang
--------

- Exposed arguments of ``clang::annotate``.

Static Analyzer
---------------

- Added a new checker ``core.BitwiseShift`` which reports situations where
  bitwise shift operators produce undefined behavior (because some operand is
  negative or too large).

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
