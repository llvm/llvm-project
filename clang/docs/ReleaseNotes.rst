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

Bug Fixes
---------
- Fixes an accepts-invalid bug in C when using a ``_Noreturn`` function
  specifier on something other than a function declaration. This fixes
  `Issue 56800 <https://github.com/llvm/llvm-project/issues/56800>`_.
- Fix `#56772 <https://github.com/llvm/llvm-project/issues/56772>`_ - invalid
  destructor names were incorrectly accepted on template classes.
- Improve compile-times with large dynamic array allocations with trivial
  constructors. This fixes
  `Issue 56774 <https://github.com/llvm/llvm-project/issues/56774>`_.
- No longer assert/miscompile when trying to make a vectorized ``_BitInt`` type
  using the ``ext_vector_type`` attribute (the ``vector_size`` attribute was
  already properly diagnosing this case).
- Fix clang not properly diagnosing the failing subexpression when chained
  binary operators are used in a ``static_assert`` expression.
- Fix a crash when evaluating a multi-dimensional array's array filler
  expression is element-dependent. This fixes
  `Issue 50601 <https://github.com/llvm/llvm-project/issues/56016>`_.
- Fixed a crash-on-valid with consteval evaluation of a list-initialized
  constructor for a temporary object. This fixes
  `Issue 55871 <https://github.com/llvm/llvm-project/issues/55871>`_.
- Fix `#57008 <https://github.com/llvm/llvm-project/issues/57008>`_ - Builtin
  C++ language extension type traits instantiated by a template with unexpected
  number of arguments cause an assertion fault.
- Fix multi-level pack expansion of undeclared function parameters.
  This fixes `Issue 56094 <https://github.com/llvm/llvm-project/issues/56094>`_.
- Fix `#57151 <https://github.com/llvm/llvm-project/issues/57151>`_.
  ``-Wcomma`` is emitted for void returning functions.
- ``-Wtautological-compare`` missed warnings for tautological comparisons
  involving a negative integer literal. This fixes
  `Issue 42918 <https://github.com/llvm/llvm-project/issues/42918>`_.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Clang will now correctly diagnose as ill-formed a constant expression where an
  enum without a fixed underlying type is set to a value outside the range of
  the enumeration's values. Due to the extended period of time this bug was
  present in major C++ implementations (including Clang), this error has the
  ability to be downgraded into a warning (via: -Wno-error=enum-constexpr-conversion)
  to provide a transition period for users. This diagnostic is expected to turn
  into an error-only diagnostic in the next Clang release. Fixes
  `Issue 50055: <https://github.com/llvm/llvm-project/issues/50055>`_.
- Clang will now check compile-time determinable string literals as format strings.
  Fixes `Issue 55805: <https://github.com/llvm/llvm-project/issues/55805>`_.
- ``-Wformat`` now recognizes ``%b`` for the ``printf``/``scanf`` family of
  functions and ``%B`` for the ``printf`` family of functions. Fixes
  `Issue 56885: <https://github.com/llvm/llvm-project/issues/56885>`_.
- ``-Wbitfield-constant-conversion`` now diagnoses implicit truncation when 1 is
  assigned to a 1-bit signed integer bitfield. This fixes
  `Issue 53253 <https://github.com/llvm/llvm-project/issues/53253>`_.
- ``-Wincompatible-function-pointer-types`` now defaults to an error in all C
  language modes. It may be downgraded to a warning with
  ``-Wno-error=incompatible-function-pointer-types`` or disabled entirely with
  ``-Wno-implicit-function-pointer-types``.
- Clang will now print more information about failed static assertions. In
  particular, simple static assertion expressions are evaluated to their
  compile-time value and printed out if the assertion fails.
- Diagnostics about uninitialized ``constexpr`` varaibles have been improved
  to mention the missing constant initializer.
- Correctly diagnose a future keyword if it exist as a keyword in the higher
  language version and specifies in which version it will be a keyword. This
  supports both c and c++ language.

Non-comprehensive list of changes in this release
-------------------------------------------------


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

New Pragmas in Clang
--------------------
- ...

Attribute Changes in Clang
--------------------------
- Added support for ``__attribute__((guard(nocf)))`` and C++-style
  ``[[clang::guard(nocf)]]``, which is equivalent to ``__declspec(guard(nocf))``
  when using the MSVC environment. This is to support enabling Windows Control
  Flow Guard checks with the ability to disable them for specific functions when
  using the MinGW environment.

Windows Support
---------------

AIX Support
-----------

C Language Changes in Clang
---------------------------

C2x Feature Support
-------------------

C++ Language Changes in Clang
-----------------------------

- Implemented DR692, DR1395 and DR1432. Use the ``-fclang-abi-compat=15`` option
  to get the old partial ordering behavior regarding packs.

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Support capturing structured bindings in lambdas
  (`P1091R3 <https://wg21.link/p1091r3>`_ and `P1381R1 <https://wg21.link/P1381R1>`).
  This fixes issues `GH52720 <https://github.com/llvm/llvm-project/issues/52720>`_,
  `GH54300 <https://github.com/llvm/llvm-project/issues/54300>`_,
  `GH54301 <https://github.com/llvm/llvm-project/issues/54301>`_,
  and `GH49430 <https://github.com/llvm/llvm-project/issues/49430>`_.
- Consider explicitly defaulted constexpr/consteval special member function
  template instantiation to be constexpr/consteval even though a call to such
  a function cannot appear in a constant expression.
  (C++14 [dcl.constexpr]p6 (CWG DR647/CWG DR1358))
- Correctly defer dependent immediate function invocations until template instantiation.
  This fixes `GH55601 <https://github.com/llvm/llvm-project/issues/55601>`_.
- Implemented "Conditionally Trivial Special Member Functions" (`P0848 <https://wg21.link/p0848r3>`_).
  Note: The handling of deleted functions is not yet compliant, as Clang
  does not implement `DR1496 <https://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1496>`_
  and `DR1734 <https://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1734>`_.
- Class member variables are now in scope when parsing a ``requires`` clause. Fixes
  `GH55216 <https://github.com/llvm/llvm-project/issues/55216>`_.

- Correctly set expression evaluation context as 'immediate function context' in
  consteval functions.
  This fixes `GH51182 <https://github.com/llvm/llvm-project/issues/51182>`


C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^

CUDA/HIP Language Changes in Clang
----------------------------------

Objective-C Language Changes in Clang
-------------------------------------

OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------

OpenMP Support in Clang
-----------------------

...

CUDA Support in Clang
---------------------

- ...

RISC-V Support in Clang
-----------------------

- ``sifive-7-rv32`` and ``sifive-7-rv64`` are no longer supported for `-mcpu`.
  Use `sifive-e76`, `sifive-s76`, or `sifive-u74` instead.

X86 Support in Clang
--------------------

- Support ``-mindirect-branch-cs-prefix`` for call and jmp to indirect thunk.

DWARF Support in Clang
----------------------

Arm and AArch64 Support in Clang
--------------------------------

Floating Point Support in Clang
-------------------------------

Internal API Changes
--------------------

Build System Changes
--------------------

AST Matchers
------------

clang-format
------------

clang-extdef-mapping
--------------------

libclang
--------

- ...

Static Analyzer
---------------

- Removed the deprecated ``-analyzer-store`` and
  ``-analyzer-opt-analyze-nested-blocks`` analyzer flags.
  ``scanbuild`` was also updated accordingly.
  Passing these flags will result in a hard error.

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
contact us on the Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_.
