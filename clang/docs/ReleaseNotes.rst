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

Potentially Breaking Changes
============================
These changes are ones which we think may surprise users when upgrading to
Clang |release| because of the opportunity they pose for disruption to existing
code bases.

- Clang will now correctly diagnose as ill-formed a constant expression where an
  enum without a fixed underlying type is set to a value outside the range of
  the enumeration's values.

  .. code-block:: c++

    enum E { Zero, One, Two, Three, Four };
    constexpr E Val1 = (E)3;  // Ok
    constexpr E Val2 = (E)7;  // Ok
    constexpr E Val3 = (E)8;  // Now diagnosed as out of the range [0, 7]
    constexpr E Val4 = (E)-1; // Now diagnosed as out of the range [0, 7]

  Due to the extended period of time this bug was present in major C++
  implementations (including Clang), this error has the ability to be
  downgraded into a warning (via: ``-Wno-error=enum-constexpr-conversion``) to
  provide a transition period for users. This diagnostic is expected to turn
  into an error-only diagnostic in the next Clang release. Fixes
  `Issue 50055 <https://github.com/llvm/llvm-project/issues/50055>`_.

- ``-Wincompatible-function-pointer-types`` now defaults to an error in all C
  language modes. It may be downgraded to a warning with
  ``-Wno-error=incompatible-function-pointer-types`` or disabled entirely with
  ``-Wno-implicit-function-pointer-types``.

  **NOTE:** We recommend that projects using configure scripts verify that the
  results do not change before/after setting
  ``-Werror=incompatible-function-pointer-types`` to avoid incompatibility with
  Clang 16.

  .. code-block:: c

    void func(const int *i);
    void other(void) {
      void (*fp)(int *) = func; // Previously a warning, now a downgradable error.
    }


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
- Fix a crash when generating code coverage information for an
  ``if consteval`` statement. This fixes
  `Issue 57377 <https://github.com/llvm/llvm-project/issues/57377>`_.
- Fix assert that triggers a crash during template name lookup when a type was
  incomplete but was not also a TagType. This fixes
  `Issue 57387 <https://github.com/llvm/llvm-project/issues/57387>`_.
- Fix a crash when emitting a concept-related diagnostic. This fixes
  `Issue 57415 <https://github.com/llvm/llvm-project/issues/57415>`_.
- Fix a crash when attempting to default a virtual constexpr non-special member
  function in a derived class. This fixes
  `Issue 57431 <https://github.com/llvm/llvm-project/issues/57431>`_
- Fix a crash where we attempt to define a deleted destructor. This fixes
  `Issue 57516 <https://github.com/llvm/llvm-project/issues/57516>`_
- Fix ``__builtin_assume_aligned`` crash when the 1st arg is array type. This fixes
  `Issue 57169 <https://github.com/llvm/llvm-project/issues/57169>`_
- Clang configuration files are now read through the virtual file system
  rather than the physical one, if these are different.
- Clang will now no longer treat a C 'overloadable' function without a prototype as
  a variadic function with the attribute.  This should make further diagnostics more
  clear.
- Fixes to builtin template emulation of regular templates.
  `Issue 42102 <https://github.com/llvm/llvm-project/issues/42102>`_
  `Issue 51928 <https://github.com/llvm/llvm-project/issues/51928>`_
- A SubstTemplateTypeParmType can now represent the pack index for a
  substitution from an expanded pack.
  `Issue 56099 <https://github.com/llvm/llvm-project/issues/56099>`_


Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Clang will now check compile-time determinable string literals as format strings.
  Fixes `Issue 55805: <https://github.com/llvm/llvm-project/issues/55805>`_.
- ``-Wformat`` now recognizes ``%b`` for the ``printf``/``scanf`` family of
  functions and ``%B`` for the ``printf`` family of functions. Fixes
  `Issue 56885: <https://github.com/llvm/llvm-project/issues/56885>`_.
- Introduced ``-Wsingle-bit-bitfield-constant-conversion``, grouped under
  ``-Wbitfield-constant-conversion``, which diagnoses implicit truncation when
  ``1`` is assigned to a 1-bit signed integer bitfield. This fixes
  `Issue 53253 <https://github.com/llvm/llvm-project/issues/53253>`_. To reduce
  potential false positives, this diagnostic will not diagnose use of the
  ``true`` macro (from ``<stdbool.h>>`) in C language mode despite the macro
  being defined to expand to ``1``.
- Clang will now print more information about failed static assertions. In
  particular, simple static assertion expressions are evaluated to their
  compile-time value and printed out if the assertion fails.
- Diagnostics about uninitialized ``constexpr`` varaibles have been improved
  to mention the missing constant initializer.
- Correctly diagnose a future keyword if it exist as a keyword in the higher
  language version and specifies in which version it will be a keyword. This
  supports both c and c++ language.
- When diagnosing multi-level pack expansions of mismatched lengths, Clang will
  now, in most cases, be able to point to the relevant outer parameter.
- ``no_sanitize("...")`` on a global variable for known but not relevant
  sanitizers is now just a warning. It now says that this will be ignored
  instead of incorrectly saying no_sanitize only applies to functions and
  methods.
- No longer mention ``reinterpet_cast`` in the invalid constant expression
  diagnostic note when in C mode.
- Clang will now give a more suitale diagnostic for declaration of block
  scope identifiers that have external/internal linkage that has an initializer.
  Fixes `Issue 57478: <https://github.com/llvm/llvm-project/issues/57478>`_.
- New analysis pass will now help preserve sugar when combining deductions, in an
  order agnostic way. This will be in effect when deducing template arguments,
  when deducing function return type from multiple return statements, for the
  conditional operator, and for most binary operations. Type sugar is combined
  in a way that strips the sugar which is different between terms, and preserves
  those which are common.
- Correctly diagnose use of an integer literal without a suffix whose
  underlying type is ``long long`` or ``unsigned long long`` as an extension in
  C89 mode . Clang previously only diagnosed if the literal had an explicit
  ``LL`` suffix.

Non-comprehensive list of changes in this release
-------------------------------------------------
- It's now possible to set the crash diagnostics directory through
  the environment variable ``CLANG_CRASH_DIAGNOSTICS_DIR``.
  The ``-fcrash-diagnostics-dir`` flag takes precedence.
- When using header modules, inclusion of a private header and violations of
  the `use-declaration rules
  <https://clang.llvm.org/docs/Modules.html#use-declaration>`_ are now
  diagnosed even when the includer is a textual header. This change can be
  temporarily reversed with ``-Xclang
  -fno-modules-validate-textual-header-includes``, but this flag will be
  removed in a future Clang release.

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
  using the MinGW environment. This attribute is only available for Windows
  targets.

- Introduced a new function attribute ``__attribute__((nouwtable))`` to suppress
  LLVM IR ``uwtable`` function attribute.

- Updated the value returned by ``__has_c_attribute(nodiscard)`` to ``202003L``
  based on the final date specified by the C2x committee draft. We already
  supported the ability to specify a message in the attribute, so there were no
  changes to the attribute behavior.

- Updated the value returned by ``__has_c_attribute(fallthrough)`` to ``201910L``
  based on the final date specified by the C2x committee draft. We previously
  used ``201904L`` (the date the proposal was seen by the committee) by mistake.
  There were no other changes to the attribute behavior.

Windows Support
---------------
- For the MinGW driver, added the options ``-mguard=none``, ``-mguard=cf`` and
  ``-mguard=cf-nochecks`` (equivalent to ``/guard:cf-``, ``/guard:cf`` and
  ``/guard:cf,nochecks`` in clang-cl) for enabling Control Flow Guard checks
  and generation of address-taken function table.

AIX Support
-----------
* When using ``-shared``, the clang driver now invokes llvm-nm to create an
  export list if the user doesn't specify one via linker flag or pass an
  alternative export control option.

C Language Changes in Clang
---------------------------
- Adjusted ``-Wformat`` warnings according to `WG14 N2562 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2562.pdf>`_.
  Clang will now consider default argument promotions in ``printf``, and remove
  unnecessary warnings. Especially ``int`` argument with specifier ``%hhd`` and
  ``%hd``.

C2x Feature Support
-------------------
- Implemented `WG14 N2662 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2662.pdf>`_,
  so the [[maybe_unused]] attribute may be applied to a label to silence an
  ``-Wunused-label`` warning.

C++ Language Changes in Clang
-----------------------------
- Implemented DR692, DR1395 and DR1432. Use the ``-fclang-abi-compat=15`` option
  to get the old partial ordering behavior regarding packs.
- Clang's default C++/ObjC++ standard is now ``gnu++17`` instead of ``gnu++14``.
  This means Clang will by default accept code using features from C++17 and
  conforming GNU extensions. Projects incompatible with C++17 can add
  ``-std=gnu++14`` to their build settings to restore the previous behaviour.

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Support capturing structured bindings in lambdas
  (`P1091R3 <https://wg21.link/p1091r3>`_ and `P1381R1 <https://wg21.link/P1381R1>`).
  This fixes issues `Issue 52720 <https://github.com/llvm/llvm-project/issues/52720>`_,
  `Issue 54300 <https://github.com/llvm/llvm-project/issues/54300>`_,
  `Issue 54301 <https://github.com/llvm/llvm-project/issues/54301>`_,
  and `Issue 49430 <https://github.com/llvm/llvm-project/issues/49430>`_.
- Consider explicitly defaulted constexpr/consteval special member function
  template instantiation to be constexpr/consteval even though a call to such
  a function cannot appear in a constant expression.
  (C++14 [dcl.constexpr]p6 (CWG DR647/CWG DR1358))
- Correctly defer dependent immediate function invocations until template instantiation.
  This fixes `Issue 55601 <https://github.com/llvm/llvm-project/issues/55601>`_.
- Implemented "Conditionally Trivial Special Member Functions" (`P0848 <https://wg21.link/p0848r3>`_).
  Note: The handling of deleted functions is not yet compliant, as Clang
  does not implement `DR1496 <https://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1496>`_
  and `DR1734 <https://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1734>`_.
- Class member variables are now in scope when parsing a ``requires`` clause. Fixes
  `Issue 55216 <https://github.com/llvm/llvm-project/issues/55216>`_.
- Correctly set expression evaluation context as 'immediate function context' in
  consteval functions.
  This fixes `Issue 51182 <https://github.com/llvm/llvm-project/issues/51182>`_.
- Fixes an assert crash caused by looking up missing vtable information on ``consteval``
  virtual functions. Fixes `Issue 55065 <https://github.com/llvm/llvm-project/issues/55065>`_.
- Skip rebuilding lambda expressions in arguments of immediate invocations.
  This fixes `Issue 56183 <https://github.com/llvm/llvm-project/issues/56183>`_,
  `Issue 51695 <https://github.com/llvm/llvm-project/issues/51695>`_,
  `Issue 50455 <https://github.com/llvm/llvm-project/issues/50455>`_,
  `Issue 54872 <https://github.com/llvm/llvm-project/issues/54872>`_,
  `Issue 54587 <https://github.com/llvm/llvm-project/issues/54587>`_.

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
- ``sifive-7-rv32`` and ``sifive-7-rv64`` are no longer supported for ``-mcpu``.
  Use ``sifive-e76``, ``sifive-s76``, or ``sifive-u74`` instead.

X86 Support in Clang
--------------------
- Support ``-mindirect-branch-cs-prefix`` for call and jmp to indirect thunk.

DWARF Support in Clang
----------------------

Arm and AArch64 Support in Clang
--------------------------------
- ``-march`` values for targeting armv2, armv2A, armv3 and armv3M have been removed.
  Their presence gave the impression that Clang can correctly generate code for
  them, which it cannot.

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
- Introduced the new function ``clang_getUnqualifiedType``, which mimics
  the behavior of ``QualType::getUnqualifiedType`` for ``CXType``.
- Introduced the new function ``clang_getNonReferenceType``, which mimics
  the behavior of ``QualType::getNonReferenceType`` for ``CXType``.

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
