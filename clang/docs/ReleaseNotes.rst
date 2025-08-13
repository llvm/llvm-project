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

C/C++ Language Potentially Breaking Changes
-------------------------------------------

- The ``__has_builtin`` function now only considers the currently active target when being used with target offloading.

C++ Specific Potentially Breaking Changes
-----------------------------------------
- For C++20 modules, the Reduced BMI mode will be the default option. This may introduce
  regressions if your build system supports two-phase compilation model but haven't support
  reduced BMI or it is a compiler bug or a bug in users code.

- Clang now correctly diagnoses during constant expression evaluation undefined behavior due to member
  pointer access to a member which is not a direct or indirect member of the most-derived object
  of the accessed object but is instead located directly in a sibling class to one of the classes
  along the inheritance hierarchy of the most-derived object as ill-formed.
  Other scenarios in which the member is not member of the most derived object were already
  diagnosed previously. (#GH150709)

  .. code-block:: c++

    struct A {};
    struct B : A {};
    struct C : A { constexpr int foo() const { return 1; } };
    constexpr A a;
    constexpr B b;
    constexpr C c;
    constexpr auto mp = static_cast<int(A::*)() const>(&C::foo);
    static_assert((a.*mp)() == 1); // continues to be rejected
    static_assert((b.*mp)() == 1); // newly rejected
    static_assert((c.*mp)() == 1); // accepted

ABI Changes in This Version
---------------------------

AST Dumping Potentially Breaking Changes
----------------------------------------
- How nested name specifiers are dumped and printed changes, keeping track of clang AST changes.

Clang Frontend Potentially Breaking Changes
-------------------------------------------

Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------
- TypeKind ``ELABORATED`` is not used anymore, per clang AST changes removing
  ElaboratedTypes. The value becomes unused, and all the existing users should
  expect the former underlying type to be reported instead.

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

C Language Changes
------------------

C2y Feature Support
^^^^^^^^^^^^^^^^^^^

C23 Feature Support
^^^^^^^^^^^^^^^^^^^

Non-comprehensive list of changes in this release
-------------------------------------------------
- Added ``__builtin_elementwise_fshl`` and ``__builtin_elementwise_fshr``.

- Added ``__builtin_elementwise_minnumnum`` and ``__builtin_elementwise_maxnumnum``.

- Trapping UBSan (e.g. ``-fsanitize-trap=undefined``) now emits a string describing the reason for
  trapping into the generated debug info. This feature allows debuggers (e.g. LLDB) to display
  the reason for trapping if the trap is reached. The string is currently encoded in the debug
  info as an artificial frame that claims to be inlined at the trap location. The function used
  for the artificial frame is an artificial function whose name encodes the reason for trapping.
  The encoding used is currently the same as ``__builtin_verbose_trap`` but might change in the future.
  This feature is enabled by default but can be disabled by compiling with
  ``-fno-sanitize-annotate-debug-info-traps``.

- ``__builtin_elementwise_max`` and ``__builtin_elementwise_min`` functions for integer types can
  now be used in constant expressions.

New Compiler Flags
------------------
- New option ``-fno-sanitize-annotate-debug-info-traps`` added to disable emitting trap reasons into the debug info when compiling with trapping UBSan (e.g. ``-fsanitize-trap=undefined``).

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
- Added a separate diagnostic group ``-Wfunction-effect-redeclarations``, for the more pedantic
  diagnostics for function effects (``[[clang::nonblocking]]`` and ``[[clang::nonallocating]]``).
  Moved the warning for a missing (though implied) attribute on a redeclaration into this group.
  Added a new warning in this group for the case where the attribute is missing/implicit on
  an override of a virtual method.
- Fixed fix-it hint for fold expressions. Clang now correctly places the suggested right
  parenthesis when diagnosing malformed fold expressions. (#GH151787)

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------
- Fix a crash when marco name is empty in ``#pragma push_macro("")`` or
  ``#pragma pop_macro("")``. (#GH149762).
- `-Wunreachable-code`` now diagnoses tautological or contradictory
  comparisons such as ``x != 0 || x != 1.0`` and ``x == 0 && x == 1.0`` on
  targets that treat ``_Float16``/``__fp16`` as native scalar types. Previously
  the warning was silently lost because the operands differed only by an implicit
  cast chain. (#GH149967).

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fix an ambiguous reference to the builtin `type_info` (available when using
  `-fms-compatibility`) with modules. (#GH38400)

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``[[nodiscard]]`` is now respected on Objective-C and Objective-C++ methods.
  (#GH141504)
- Using ``[[gnu::cleanup(some_func)]]`` where some_func is annotated with
  ``[[gnu::error("some error")]]`` now correctly triggers an error. (#GH146520)

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^
<<<<<<< HEAD
- Diagnose binding a reference to ``*nullptr`` during constant evaluation. (#GH48665)
- Suppress ``-Wdeprecated-declarations`` in implicitly generated functions. (#GH147293)
- Fix a crash when deleting a pointer to an incomplete array (#GH150359).
- Fix an assertion failure when expression in assumption attribute
  (``[[assume(expr)]]``) creates temporary objects.
- Fix the dynamic_cast to final class optimization to correctly handle
  casts that are guaranteed to fail (#GH137518).
- Fix bug rejecting partial specialization of variable templates with auto NTTPs (#GH118190).
=======

- Clang now supports implicitly defined comparison operators for friend declarations. (#GH132249)
- Clang now diagnoses copy constructors taking the class by value in template instantiations. (#GH130866)
- Clang is now better at keeping track of friend function template instance contexts. (#GH55509)
- Clang now prints the correct instantiation context for diagnostics suppressed
  by template argument deduction.
- Errors that occur during evaluation of certain type traits and builtins are
  no longer incorrectly emitted when they are used in an SFINAE context. The
  type traits are:

  - ``__is_constructible`` and variants,
  - ``__is_convertible`` and variants,
  - ``__is_assignable`` and variants,
  - ``__reference_binds_to_temporary``,
    ``__reference_constructs_from_temporary``,
    ``__reference_converts_from_temporary``,
  - ``__is_trivially_equality_comparable``.

  The builtin is ``__builtin_common_type``. (#GH132044)
- Clang is now better at instantiating the function definition after its use inside
  of a constexpr lambda. (#GH125747)
- Fixed a local class member function instantiation bug inside dependent lambdas. (#GH59734), (#GH132208)
- Clang no longer crashes when trying to unify the types of arrays with
  certain differences in qualifiers (this could happen during template argument
  deduction or when building a ternary operator). (#GH97005)
- Fixed type alias CTAD issues involving default template arguments. (#GH134471)
- Fixed CTAD issues when initializing anonymous fields with designated initializers. (#GH67173)
- The initialization kind of elements of structured bindings
  direct-list-initialized from an array is corrected to direct-initialization.
- Clang no longer crashes when a coroutine is declared ``[[noreturn]]``. (#GH127327)
- Clang now uses the parameter location for abbreviated function templates in ``extern "C"``. (#GH46386)
- Clang will emit an error instead of crash when use co_await or co_yield in
  C++26 braced-init-list template parameter initialization. (#GH78426)
- Improved fix for an issue with pack expansions of type constraints, where this
  now also works if the constraint has non-type or template template parameters.
  (#GH131798)
- Fixes to partial ordering of non-type template parameter packs. (#GH132562)
- Fix crash when evaluating the trailing requires clause of generic lambdas which are part of
  a pack expansion.
- Fixes matching of nested template template parameters. (#GH130362)
- Correctly diagnoses template template parameters which have a pack parameter
  not in the last position.
- Disallow overloading on struct vs class on dependent types, which is IFNDR, as
  this makes the problem diagnosable.
- Improved preservation of the presence or absence of typename specifier when
  printing types in diagnostics.
- Clang now correctly parses ``if constexpr`` expressions in immediate function context. (#GH123524)
- Fixed an assertion failure affecting code that uses C++23 "deducing this". (#GH130272)
- Clang now properly instantiates destructors for initialized members within non-delegating constructors. (#GH93251)
- Correctly diagnoses if unresolved using declarations shadows template parameters (#GH129411)
- Fixed C++20 aggregate initialization rules being incorrectly applied in certain contexts. (#GH131320)
- Clang was previously coalescing volatile writes to members of volatile base class subobjects.
  The issue has been addressed by propagating qualifiers during derived-to-base conversions in the AST. (#GH127824)
- Correctly propagates the instantiated array type to the ``DeclRefExpr`` that refers to it. (#GH79750), (#GH113936), (#GH133047)
- Fixed a Clang regression in C++20 mode where unresolved dependent call expressions were created inside non-dependent contexts (#GH122892)
- Clang now emits the ``-Wunused-variable`` warning when some structured bindings are unused
  and the ``[[maybe_unused]]`` attribute is not applied. (#GH125810)
- Declarations using class template argument deduction with redundant
  parentheses around the declarator are no longer rejected. (#GH39811)
- Fixed a crash caused by invalid declarations of ``std::initializer_list``. (#GH132256)
- Clang no longer crashes when establishing subsumption between some constraint expressions. (#GH122581)
- Clang now issues an error when placement new is used to modify a const-qualified variable
  in a ``constexpr`` function. (#GH131432)
- Fixed an incorrect TreeTransform for calls to ``consteval`` functions if a conversion template is present. (#GH137885)
- Clang now emits a warning when class template argument deduction for alias templates is used in C++17. (#GH133806)
- Fixed a missed initializer instantiation bug for variable templates. (#GH134526), (#GH138122)
- Fix a crash when checking the template template parameters of a dependent lambda appearing in an alias declaration.
  (#GH136432), (#GH137014), (#GH138018)
- Fixed an assertion when trying to constant-fold various builtins when the argument
  referred to a reference to an incomplete type. (#GH129397)
- Fixed a crash when a cast involved a parenthesized aggregate initialization in dependent context. (#GH72880)
- No longer crashes when instantiating invalid variable template specialization
  whose type depends on itself. (#GH51347), (#GH55872)
- Improved parser recovery of invalid requirement expressions. In turn, this
  fixes crashes from follow-on processing of the invalid requirement. (#GH138820)
- Fixed the handling of pack indexing types in the constraints of a member function redeclaration. (#GH138255)
- Clang now correctly parses arbitrary order of ``[[]]``, ``__attribute__`` and ``alignas`` attributes for declarations (#GH133107)
- Fixed a crash when forming an invalid function type in a dependent context. (#GH138657) (#GH115725) (#GH68852)
- Clang no longer segfaults when there is a configuration mismatch between modules and their users (http://crbug.com/400353616).
- Fix an incorrect deduction when calling an explicit object member function template through an overload set address.
- Fixed bug in constant evaluation that would allow using the value of a
  reference in its own initializer in C++23 mode (#GH131330).
- Clang could incorrectly instantiate functions in discarded contexts (#GH140449)
- Fix instantiation of default-initialized variable template specialization. (#GH140632) (#GH140622)
- Clang modules now allow a module and its user to differ on TrivialAutoVarInit*
- Fixed an access checking bug when initializing non-aggregates in default arguments (#GH62444), (#GH83608)
- Fixed a pack substitution bug in deducing class template partial specializations. (#GH53609)
- Fixed a crash when constant evaluating some explicit object member assignment operators. (#GH142835)
>>>>>>> 14b0e97a546e (Fix more tests in error handling (#53))

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fix incorrect name qualifiers applied to alias CTAD. (#GH136624)
- Fixed ElaboratedTypes appearing within NestedNameSpecifier, which was not a
  legal representation. This is fixed because ElaboratedTypes don't exist anymore. (#GH43179) (#GH68670) (#GH92757)

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

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

X86 Support
^^^^^^^^^^^

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

- Add support for `__attribute__((interrupt("rnmi")))` to be used with the `Smrnmi` extension.
  With this the `Smrnmi` extension is fully supported.

- Add `-march=unset` to clear any previous `-march=` value. This ISA string will
  be computed from `-mcpu` or the platform default.

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
- Removed elaboratedType matchers, and related nested name specifier changes,
  following the corresponding changes in the clang AST.
- Ensure ``hasBitWidth`` doesn't crash on bit widths that are dependent on template
  parameters.

clang-format
------------

libclang
--------

Code Completion
---------------

Static Analyzer
---------------
- The Clang Static Analyzer now handles parenthesized initialization.
  (#GH148875)
- ``__datasizeof`` (C++) and ``_Countof`` (C) no longer cause a failed assertion
  when given an operand of VLA type. (#GH151711)

New features
^^^^^^^^^^^^

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^
- Fixed a crash in the static analyzer that when the expression in an
  ``[[assume(expr)]]`` attribute was enclosed in parentheses.  (#GH151529)
- Fixed a crash when parsing ``#embed`` parameters with unmatched closing brackets. (#GH152829)

Improvements
^^^^^^^^^^^^

Moved checkers
^^^^^^^^^^^^^^

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------
- Exposed `clang_getCursorLanguage` via `Cursor.language`.

OpenMP Support
--------------
- Added parsing and semantic analysis support for the ``need_device_addr``
  modifier in the ``adjust_args`` clause.
- Allow array length to be omitted in array section subscript expression.
- Fixed non-contiguous strided update in the ``omp target update`` directive with the ``from`` clause.

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
