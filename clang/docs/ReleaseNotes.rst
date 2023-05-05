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
- Support for out-of-line definitions of constrained templates has been improved.
  This partially fixes `#49620 <https://github.com/llvm/llvm-project/issues/49620>`_.
- Lambda templates with a requires clause directly after the template parameters now parse
  correctly if the requires clause consists of a variable with a dependent type.
  (`#61278 <https://github.com/llvm/llvm-project/issues/61278>`_)
- Announced C++20 Coroutines is fully supported on all targets except Windows, which
  still has some stability and ABI issues.
- Downgraded use of a reserved identifier in a module export declaration from
  an error to a warning under the ``-Wreserved-module-identifier`` warning
  group. This warning is enabled by default. This addresses `#61446
  <https://github.com/llvm/llvm-project/issues/61446>`_ and allows easier
  building of standard modules. This diagnostic may be strengthened into an
  error again in the future once there is a less fragile way to mark a module
  as being part of the implementation rather than a user module.

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P2036R3: Change scope of lambda trailing-return-type <https://wg21.link/P2036R3>`_
  and `P2579R0 Mitigation strategies for P2036 <https://wg21.link/P2579R0>`_.
  These proposals modify how variables captured in lambdas can appear in trailing return type
  expressions and how their types are deduced therein, in all C++ language versions.
- Implemented partial support for `P2448R2: Relaxing some constexpr restrictions <https://wg21.link/p2448r2>`_
  Explicitly defaulted functions no longer have to be constexpr-compatible but merely constexpr suitable.
  We do not support outside of defaulted special memeber functions the change that constexpr functions no
  longer have to be constexpr compatible but rather support a less restricted requirements for constexpr
  functions. Which include allowing non-literal types as return values and paremeters, allow calling of
  non-constexpr functions and constructors.

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Implemented `DR2397 <https://wg21.link/CWG2397>`_ which allows ``auto`` specifier for pointers
  and reference to arrays.

C Language Changes
------------------
- Support for outputs from asm goto statements along indirect edges has been
  added. (`#53562 <https://github.com/llvm/llvm-project/issues/53562>`_)
- Fixed a bug that prevented initialization of an ``_Atomic``-qualified pointer
  from a null pointer constant.
- Fixed a bug that prevented casting to an ``_Atomic``-qualified type.
  (`#39596 <https://github.com/llvm/llvm-project/issues/39596>`_)

C2x Feature Support
^^^^^^^^^^^^^^^^^^^
- Implemented the ``unreachable`` macro in freestanding ``<stddef.h>`` for
  `WG14 N2826 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2826.pdf>`_

- Removed the ``ATOMIC_VAR_INIT`` macro in C2x and later standards modes, which
  implements `WG14 N2886 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2886.htm>`_

- Implemented `WG14 N2934 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2934.pdf>`_
  which introduces the ``bool``, ``static_assert``, ``alignas``, ``alignof``,
  and ``thread_local`` keywords in C2x.

- Implemented `WG14 N2900 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2900.htm>`_
  and `WG14 N3011 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3011.htm>`_
  which allows for empty braced initialization in C.

  .. code-block:: c

    struct S { int x, y } s = {}; // Initializes s.x and s.y to 0

  As part of this change, the ``-Wgnu-empty-initializer`` warning group was
  removed, as this is no longer a GNU extension but a C2x extension. You can
  use ``-Wno-c2x-extensions`` to silence the extension warning instead.

- Updated the implementation of
  `WG14 N3042 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3042.htm>`_
  based on decisions reached during the WG14 CD Ballot Resolution meetings held
  in Jan and Feb 2023. This should complete the implementation of ``nullptr``
  and ``nullptr_t`` in C. The specific changes are:

  .. code-block:: c

    void func(nullptr_t);
    func(0); // Previously required to be rejected, is now accepted.
    func((void *)0); // Previously required to be rejected, is now accepted.

    nullptr_t val;
    val = 0; // Previously required to be rejected, is now accepted.
    val = (void *)0; // Previously required to be rejected, is now accepted.

    bool b = nullptr; // Was incorrectly rejected by Clang, is now accepted.


Non-comprehensive list of changes in this release
-------------------------------------------------
- Clang now saves the address of ABI-indirect function parameters on the stack,
  improving the debug information available in programs compiled without
  optimizations.
- Clang now supports ``__builtin_nondeterministic_value`` that returns a
  nondeterministic value of the same type as the provided argument.
- Clang now supports ``__builtin_FILE_NAME()`` which returns the same
  information as the ``__FILE_NAME__`` macro (the presumed file name
  from the invocation point, with no path components included).
- Clang now supports ``__builtin_assume_separate_storage`` that indicates that
  its arguments point to objects in separate storage allocations.
- Clang now supports expressions in ``#pragma clang __debug dump``.
- Clang now supports declaration of multi-dimensional arrays with
  ``__declspec(property)``.
- A new builtin type trait ``__is_trivially_equaltiy_comparable`` has been added,
  which checks whether comparing two instances of a type is equivalent to
  ``memcmp(&lhs, &rhs, sizeof(T)) == 0``.
- Clang now ignores null directives outside of the include guard when deciding
  whether a file can be enabled for the multiple-include optimization.

New Compiler Flags
------------------
- The flag ``-std=c++23`` has been added. This behaves the same as the existing
  flag ``-std=c++2b``.

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
- The CodeGen flag `-lower-global-dtors-via-cxa-atexit` which affects how global
  destructors are lowered for MachO is removed without replacement. The default
  of `-lower-global-dtors-via-cxa-atexit=true` is now the only supported way.

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
- The ``__has_attribute``, ``__has_c_attribute`` and ``__has_cpp_attribute``
  preprocessor operators now return 1 also for attributes defined by plugins.

Improvements to Clang's diagnostics
-----------------------------------
- We now generate a diagnostic for signed integer overflow due to unary minus
  in a non-constant expression context.
  (`#31643 <https://github.com/llvm/llvm-project/issues/31643>`_)
- Clang now warns by default for C++20 and later about deprecated capture of
  ``this`` with a capture default of ``=``. This warning can be disabled with
  ``-Wno-deprecated-this-capture``.
- Clang had failed to emit some ``-Wundefined-internal`` for members of a local
  class if that class was first introduced with a forward declaration.
- Diagnostic notes and fix-its are now generated for ``ifunc``/``alias`` attributes
  which point to functions whose names are mangled.
- Diagnostics relating to macros on the command line of a preprocessed assembly
  file or precompiled header are now reported as coming from the file
  ``<command line>`` instead of ``<built-in>``.
- Clang constexpr evaluator now provides a more concise diagnostic when calling
  function pointer that is known to be null.
- Clang now avoids duplicate warnings on unreachable ``[[fallthrough]];`` statements
  previously issued from ``-Wunreachable-code`` and ``-Wunreachable-code-fallthrough``
  by prioritizing ``-Wunreachable-code-fallthrough``.
- Clang now correctly diagnoses statement attributes ``[[clang::always_inine]]`` and
  ``[[clang::noinline]]`` when used on a statement with dependent call expressions.
- Clang now checks for completeness of the second and third arguments in the
  conditional operator.
  (`#59718 <https://github.com/llvm/llvm-project/issues/59718>`_)
- There were some cases in which the diagnostic for the unavailable attribute
  might not be issued, this fixes those cases.
  (`61815 <https://github.com/llvm/llvm-project/issues/61815>`_)
- Clang now avoids unnecessary diagnostic warnings for obvious expressions in
  the case of binary operators with logical OR operations.
  (`#57906 <https://github.com/llvm/llvm-project/issues/57906>`_)
- Clang's "static assertion failed" diagnostic now points to the static assertion
  expression instead of pointing to the ``static_assert`` token.
  (`#61951 <https://github.com/llvm/llvm-project/issues/61951>`_)
- ``-Wformat`` now recognizes ``%lb`` for the ``printf``/``scanf`` family of
  functions.
  (`#62247: <https://github.com/llvm/llvm-project/issues/62247>`_).
- Clang now diagnoses shadowing of lambda's template parameter by a capture.
  (`#61105: <https://github.com/llvm/llvm-project/issues/61105>`_).

Bug Fixes in This Version
-------------------------

- Fix segfault while running clang-rename on a non existing file.
  (`#36471 <https://github.com/llvm/llvm-project/issues/36471>`_)
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
- No longer issue a false positive diagnostic about a catch handler that cannot
  be reached despite being reachable. This fixes
  `#61177 <https://github.com/llvm/llvm-project/issues/61177>`_ in anticipation
  of `CWG2699 <https://wg21.link/CWG2699>_` being accepted by WG21.
- Fix crash when parsing fold expression containing a delayed typo correction.
  (`#61326 <https://github.com/llvm/llvm-project/issues/61326>`_)
- Fix crash when dealing with some member accesses outside of class or member
  function context.
  (`#37792 <https://github.com/llvm/llvm-project/issues/37792>`_) and
  (`#48405 <https://github.com/llvm/llvm-project/issues/48405>`_)
- Fix crash when using ``[[clang::always_inline]]`` or ``[[clang::noinline]]``
  statement attributes on a call to a template function in the body of a
  template function.
- Fix coroutines issue where ``get_return_object()`` result was always eargerly
  converted to the return type. Eager initialization (allowing RVO) is now only
  perfomed when these types match, otherwise deferred initialization is used,
  enabling short-circuiting coroutines use cases. This fixes
  (`#56532 <https://github.com/llvm/llvm-project/issues/56532>`_) in
  antecipation of `CWG2563 <https://cplusplus.github.io/CWG/issues/2563.html>_`.
- Fix highlighting issue with ``_Complex`` and initialization list with more than
  2 items. (`#61518 <https://github.com/llvm/llvm-project/issues/61518>`_)
- Fix  ``getSourceRange`` on  ``VarTemplateSpecializationDecl`` and
  ``VarTemplatePartialSpecializationDecl``, which represents variable with
  the initializer, so it behaves consistently with other ``VarDecls`` and ends
  on the last token of initializer, instead of right angle bracket of
  the template argument list.
- Fix false-positive diagnostic issued for consteval initializers of temporary
  objects.
  (`#60286 <https://github.com/llvm/llvm-project/issues/60286>`_)
- Correct restriction of trailing requirements clauses on a templated function.
  Previously we only rejected non-'templated' things, but the restrictions ALSO need
  to limit non-defined/non-member functions as well. Additionally, we now diagnose
  requires on lambdas when not allowed, which we previously missed.
  (`#61748 <https://github.com/llvm/llvm-project/issues/61748>`_)
- Fix confusing diagnostic for incorrect use of qualified concepts names.
- Fix handling of comments in function like macros so they are ignored in -CC
  mode.
  (`#60887 <https://github.com/llvm/llvm-project/issues/60887>`_)
- Fix incorrect merging of lambdas across modules.
  (`#60985 <https://github.com/llvm/llvm-project/issues/60985>`_)
- Fix crash when handling nested immediate invocations in initializers of global
  variables.
  (`#58207 <https://github.com/llvm/llvm-project/issues/58207>`_)
- Fix crash when generating code coverage information for `PseudoObjectExpr` in
  Clang AST.
  (`#45481 <https://github.com/llvm/llvm-project/issues/45481>`_)
- Fix the assertion hit when a template consteval function appears in a nested
  consteval/constexpr call chain.
  (`#61142 <https://github.com/llvm/llvm-project/issues/61142>`_)
- Clang now better diagnose placeholder types constrained with a concept that is
  not a type concept.
- Fix crash when a doc comment contains a line splicing.
  (`#62054 <https://github.com/llvm/llvm-project/issues/62054>`_)
- Work around with a clang coverage crash which happens when visiting
  expressions/statements with invalid source locations in non-assert builds.
  Assert builds may still see assertions triggered from this.
- Fix a failed assertion due to an invalid source location when trying to form
  a coverage report for an unresolved constructor expression.
  (`#62105 <https://github.com/llvm/llvm-project/issues/62105>`_)
- Fix defaulted equality operator so that it does not attempt to compare unnamed
  bit-fields. This fixes:
  (`#61355 <https://github.com/llvm/llvm-project/issues/61335>`_) and
  (`#61417 <https://github.com/llvm/llvm-project/issues/61417>`_)
- Fix crash after suggesting typo correction to constexpr if condition.
  (`#61885 <https://github.com/llvm/llvm-project/issues/61885>`_)
- Clang constexpr evaluator now treats comparison of [[gnu::weak]]-attributed
  member pointer as an invalid expression.
- Fix crash when member function contains invalid default argument.
  (`#62122 <https://github.com/llvm/llvm-project/issues/62122>`_)
- Fix crash when handling undefined template partial specialization
  (`#61356 <https://github.com/llvm/llvm-project/issues/61356>`_)
- Fix premature substitution into the constraints of an inherited constructor.
- Fix crash when attempting to perform parenthesized initialization of an
  aggregate with a base class with only non-public constructors.
  (`#62296 <https://github.com/llvm/llvm-project/issues/62296>`_)
- Fix a stack overflow issue when evaluating ``consteval`` default arguments.
  (`#60082` <https://github.com/llvm/llvm-project/issues/60082>`_)
- Fix the assertion hit when generating code for global variable initializer of
  _BitInt(1) type.
  (`#62207 <https://github.com/llvm/llvm-project/issues/62207>`_)
- Fix lambdas and other anonymous function names not respecting ``-fdebug-prefix-map``
  (`#62192 <https://github.com/llvm/llvm-project/issues/62192>`_)
- Fix crash when attempting to pass a non-pointer type as first argument of
  ``__builtin_assume_aligned``.
  (`#62305 <https://github.com/llvm/llvm-project/issues/62305>`_) 

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed a bug where attribute annotations on type specifiers (enums, classes,
  structs, unions, and scoped enums) were not properly ignored, resulting in
  misleading warning messages. Now, such attribute annotations are correctly
  ignored. (`#61660 <https://github.com/llvm/llvm-project/issues/61660>`_)

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Fix crash on invalid code when looking up a destructor in a templated class
  inside a namespace.
  (`#59446 <https://github.com/llvm/llvm-project/issues/59446>`_)
- Fix crash when evaluating consteval constructor of derived class whose base
  has more than one field.
  (`#60166 <https://github.com/llvm/llvm-project/issues/60166>`_)
- Fix an issue about ``decltype`` in the members of class templates derived from
  templates with related parameters.
  (`#58674 <https://github.com/llvm/llvm-project/issues/58674>`_)
- Fix incorrect deletion of the default constructor of unions in some
  cases. (`#48416 <https://github.com/llvm/llvm-project/issues/48416>`_)
- No longer issue a pre-C++23 compatibility warning in ``-pedantic`` mode
  regading overloaded `operator[]` with more than one parmeter or for static
  lambdas. (`#61582 <https://github.com/llvm/llvm-project/issues/61582>`_)
- Stop stripping CV qualifiers from the type of ``this`` when capturing it by value in
  a lambda.
  (`#50866 <https://github.com/llvm/llvm-project/issues/50866>`_)
- Fix ordering of function templates by constraints when they have template
  template parameters with different nested constraints.
- Fix type equivalence comparison between auto types to take constraints into
  account.
- Fix bug in the computation of the ``__has_unique_object_representations``
  builtin for types with unnamed bitfields.
  (`#61336 <https://github.com/llvm/llvm-project/issues/61336>`_)
- Fix default member initializers sometimes being ignored when performing
  parenthesized aggregate initialization of templated types.
  (`#62266 <https://github.com/llvm/llvm-project/issues/62266>`_)
- Fix overly aggressive lifetime checks for parenthesized aggregate
  initialization.
  (`#61567 <https://github.com/llvm/llvm-project/issues/61567>`_)
- Fix a crash when expanding a pack as the index of a subscript expression.

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Dumping the AST to JSON no longer causes a failed assertion when targetting
  the Microsoft ABI and the AST to be dumped contains dependent names that
  would not typically be mangled.
  (`#61440 <https://github.com/llvm/llvm-project/issues/61440>`_)

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

- Linking for AMDGPU now uses ``--no-undefined`` by default. This causes
  undefined symbols in the created module to be a linker error. To prevent this,
  pass ``-Wl,--undefined`` if compiling directly, or ``-Xoffload-linker
  --undefined`` if using an offloading language.
- The deprecated ``-mcode-object-v3`` and ``-mno-code-object-v3`` command-line
  options have been removed.

X86 Support
^^^^^^^^^^^

- Add ISA of ``AMX-COMPLEX`` which supports ``tcmmimfp16ps`` and
  ``tcmmrlfp16ps``.

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

- Clang supports flag output operands by which conditions in the NZCV could be outputs
  of inline assembly for AArch64. This change is more consistent with the behavior of
  GCC.

   .. code-block:: c

     // int a = foo(); int* b = bar();
     asm("ands %w[a], %w[a], #3" : [a] "+r"(a), "=@cceq"(*b));

- Fix a crash when ``preserve_all`` calling convention is used on AArch64.
  `Issue 58145 <https://github.com/llvm/llvm-project/issues/58145>`_

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

- Patchable function entry (``-fpatchable-function-entry``) is now supported
  on LoongArch.

RISC-V Support
^^^^^^^^^^^^^^
- Added ``-mrvv-vector-bits=`` option to give an upper and lower bound on vector
  length. Valid values are powers of 2 between 64 and 65536. A value of 32
  should eventually be supported. We also accept "zvl" to use the Zvl*b
  extension from ``-march`` or ``-mcpu`` to the be the upper and lower bound.
- Fixed incorrect ABI lowering of ``_Float16`` in the case of structs
  containing ``_Float16`` that are eligible for passing via GPR+FPR or
  FPR+FPR.
- Removed support for ``__attribute__((interrupt("user")))``. User-level
  interrupts are not in version 1.12 of the privileged specification.
- Added ``attribute(riscv_rvv_vector_bits(__riscv_v_fixed_vlen))`` to allow
  the size of a RVV (RISC-V Vector) scalable type to be specified. This allows
  RVV scalable vector types to be used in structs or in global variables.

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA Support
^^^^^^^^^^^^

AIX Support
^^^^^^^^^^^
- Add an AIX-only link-time option, `-mxcoff-build-id=0xHEXSTRING`, to allow users
  to embed a hex id in their binary such that it's readable by the program itself.
  This option is an alternative to the `--build-id=0xHEXSTRING` GNU linker option
  which is currently not supported by the AIX linker.


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
- Add ``__builtin_set_flt_rounds`` builtin for X86, x86_64, Arm and AArch64 only.

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
- Add additional Qualifier Ordering support for special cases such
  as templates, requires clauses, long qualified names.
- Fix all known issues associated with ``LambdaBodyIndentation: OuterScope``.
- Add ``BracedInitializerIndentWidth`` which can be used to configure
  the indentation level of the contents of braced init lists.

libclang
--------

- Introduced the new function ``clang_CXXMethod_isExplicit``,
  which identifies whether a constructor or conversion function cursor
  was marked with the explicit identifier.

- Introduced the new ``CXIndex`` constructor function
  ``clang_createIndexWithOptions``, which allows storing precompiled preambles
  in memory or overriding the precompiled preamble storage path.

- Deprecated two functions ``clang_CXIndex_setGlobalOptions`` and
  ``clang_CXIndex_setInvocationEmissionPathOption`` in favor of the new
  function ``clang_createIndexWithOptions`` in order to improve thread safety.

- Added check in ``clang_getFieldDeclBitWidth`` for whether a bit-field
  has an evaluable bit width. Fixes undefined behavior when called on a
  bit-field whose width depends on a template paramter.

Static Analyzer
---------------
- Fix incorrect alignment attribute on the this parameter of certain
  non-complete destructors when using the Microsoft ABI.
  (`#60465 <https://github.com/llvm/llvm-project/issues/60465>`_)

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------
The following methods have been added:

- ``clang_Location_isInSystemHeader`` exposed via the ``is_in_system_header``
  property of the `Location` class.

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
