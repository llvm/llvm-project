.. If you want to modify sections/contents permanently, you should modify both
   ReleaseNotes.rst and ReleaseNotesTemplate.txt.

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

- Clang will now emit a warning if the auto-detected GCC installation
  directory (i.e. the one with the largest version number) does not
  contain libstdc++ include directories although a "complete" GCC
  installation directory containing the include directories is
  available. It is planned to change the auto-detection to prefer the
  "complete" directory in the future.  The warning will disappear if
  the libstdc++ include directories are either installed or removed
  for all GCC installation directories considered by the
  auto-detection; see the output of ``clang -v`` for a list of those
  directories. If the GCC installations cannot be modified and
  maintaining the current choice of the auto-detection is desired, the
  GCC installation directory can be selected explicitly using the
  ``--gcc-install-dir`` command line argument. This will silence the
  warning. It can also be disabled using the
  ``-Wno-gcc-install-dir-libstdcxx`` command line flag.
- Scalar deleting destructor support has been aligned with MSVC when
  targeting the MSVC ABI. Clang previously implemented support for
  ``::delete`` by calling the complete object destructor and then the
  appropriate global delete operator (as is done for the Itanium ABI).
  The scalar deleting destructor is now called to destroy the object
  and deallocate its storage. This is an ABI change that can result in
  memory corruption when a program built for the MSVC ABI has
  portions compiled with clang 21 or earlier and portions compiled
  with a version of clang 22 (or MSVC). Consider a class ``X`` that
  declares a virtual destructor and an ``operator delete`` member
  with the destructor defined in library ``A`` and a call to `::delete`` in
  library ``B``. If library ``A`` is compiled with clang 21 and library ``B``
  is compiled with clang 22, the ``::delete`` call might dispatch to the
  scalar deleting destructor emitted in library ``A`` which will erroneously
  call the member ``operator delete`` instead of the expected global
  delete operator. The old behavior is retained under ``-fclang-abi-compat=21``
  flag.

C/C++ Language Potentially Breaking Changes
-------------------------------------------

- The ``__has_builtin`` function now only considers the currently active target when being used with target offloading.

- The ``-Wincompatible-pointer-types`` diagnostic now defaults to an error;
  it can still be downgraded to a warning by passing ``-Wno-error=incompatible-pointer-types``. (#GH74605)

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

- ``VarTemplateSpecializationDecl::getTemplateArgsAsWritten()`` method now
  returns ``nullptr`` for implicitly instantiated declarations.

ABI Changes in This Version
---------------------------

AST Dumping Potentially Breaking Changes
----------------------------------------
- How nested name specifiers are dumped and printed changes, keeping track of clang AST changes.

- Pretty-printing of atomic builtins ``__atomic_test_and_set`` and ``__atomic_clear`` in ``-ast-print`` output.
  These previously displayed an extra ``<null expr>`` argument, e.g.:

    ``__atomic_test_and_set(p, <null expr>, 0)``

  Now they are printed as:

    ``__atomic_test_and_set(p, 0)``

- Pretty-printing of templates with inherited (i.e. specified in a previous
  redeclaration) default arguments has been fixed.

- Default arguments of template template parameters are pretty-printed now.

Clang Frontend Potentially Breaking Changes
-------------------------------------------
- Members of anonymous unions/structs are now injected as ``IndirectFieldDecl``
  into the enclosing record even if their names conflict with other names in the
  scope. These ``IndirectFieldDecl`` are marked invalid.

Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------
- Return ``None`` instead of null cursors from ``Token.cursor``
- TypeKind ``ELABORATED`` is not used anymore, per clang AST changes removing
  ElaboratedTypes. The value becomes unused, and all the existing users should
  expect the former underlying type to be reported instead.
- Remove ``AccessSpecifier.NONE`` kind. No libclang interfaces ever returned this kind.

What's New in Clang |release|?
==============================

C++ Language Changes
--------------------

- A new family of builtins ``__builtin_*_synthesizes_from_spaceship`` has been added. These can be queried to know
  whether the ``<`` (``lt``), ``>`` (``gt``), ``<=`` (``le``), or ``>=`` (``ge``) operators are synthesized from a
  ``<=>``. This makes it possible to optimize certain facilities by using the ``<=>`` operation directly instead of
  doing multiple comparisons.

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Started the implementation of `P2686R5 <https://wg21.link/P2686R5>`_ Constexpr structured bindings.
  At this timem, references to constexpr and decomposition of *tuple-like* types are not supported
  (only arrays and aggregates are).

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Clang now normalizes constraints before checking whether they are satisfied, as mandated by the standard.
  As a result, Clang no longer incorrectly diagnoses substitution failures in template arguments only
  used in concept-ids, and produces better diagnostics for satisfaction failure. (#GH61811) (#GH135190)

C++17 Feature Support
^^^^^^^^^^^^^^^^^^^^^

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C Language Changes
------------------

C2y Feature Support
^^^^^^^^^^^^^^^^^^^
- No longer triggering ``-Wstatic-in-inline`` in C2y mode; use of a static
  function or variable within an extern inline function is no longer a
  constraint per `WG14 N3622 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3622.txt>`_.
- Clang now supports `N3355 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3355.htm>`_ Named Loops.

C23 Feature Support
^^^^^^^^^^^^^^^^^^^
- Added ``FLT_SNAN``, ``DBL_SNAN``, and ``LDBL_SNAN`` to Clang's ``<float.h>``
  header in C23 and later modes. This implements
  `WG14 N2710 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2710.htm>`_.

Non-comprehensive list of changes in this release
-------------------------------------------------
- Added ``__builtin_elementwise_fshl`` and ``__builtin_elementwise_fshr``.

- ``__builtin_elementwise_abs`` can now be used in constant expression.

- Added ``__builtin_elementwise_minnumnum`` and ``__builtin_elementwise_maxnumnum``.

- Trapping UBSan (e.g. ``-fsanitize=undefined -fsanitize-trap=undefined``) now
  emits a string describing the reason for trapping into the generated debug
  info. This feature allows debuggers (e.g. LLDB) to display the reason for
  trapping if the trap is reached. The string is currently encoded in the debug
  info as an artificial frame that claims to be inlined at the trap location.
  The function used for the artificial frame is an artificial function whose
  name encodes the reason for trapping. The encoding used is currently the same
  as ``__builtin_verbose_trap`` but might change in the future. This feature is
  enabled by default but can be disabled by compiling with
  ``-fno-sanitize-debug-trap-reasons``. The feature has a ``basic`` and
  ``detailed`` mode (the default). The ``basic`` mode emits a hard-coded string
  per trap kind (e.g. ``Integer addition overflowed``) and the ``detailed`` mode
  emits a more descriptive string describing each individual trap (e.g. ``signed
  integer addition overflow in 'a + b'``). The ``detailed`` mode produces larger
  debug info than ``basic`` but is more helpful for debugging. The
  ``-fsanitize-debug-trap-reasons=`` flag can be used to switch between the
  different modes or disable the feature entirely. Note due to trap merging in
  optimized builds (i.e. in each function all traps of the same kind get merged
  into the same trap instruction) the trap reasons might be removed. To prevent
  this build without optimizations (i.e. use `-O0` or use the `optnone` function
  attribute) or use the `fno-sanitize-merge=` flag in optimized builds.

- ``__builtin_elementwise_max`` and ``__builtin_elementwise_min`` functions for integer types can
  now be used in constant expressions.

- A vector of booleans is now a valid condition for the ternary ``?:`` operator.
  This binds to a simple vector select operation.

- Added ``__builtin_masked_load``, ``__builtin_masked_expand_load``,
  ``__builtin_masked_store``, ``__builtin_masked_compress_store`` for
  conditional memory loads from vectors. Binds to the LLVM intrinsics of the
  same name.

- Added ``__builtin_masked_gather`` and ``__builtin_masked_scatter`` for
  conditional gathering and scattering operations on vectors. Binds to the LLVM
  intrinsics of the same name.

- The ``__builtin_popcountg``, ``__builtin_ctzg``, and ``__builtin_clzg``
  functions now accept fixed-size boolean vectors.

- Use of ``__has_feature`` to detect the ``ptrauth_qualifier`` and ``ptrauth_intrinsics``
  features has been deprecated, and is restricted to the arm64e target only. The
  correct method to check for these features is to test for the ``__PTRAUTH__``
  macro.

- Added a new builtin, ``__builtin_dedup_pack``, to remove duplicate types from a parameter pack.
  This feature is particularly useful in template metaprogramming for normalizing type lists.
  The builtin produces a new, unexpanded parameter pack that can be used in contexts like template
  argument lists or base specifiers.

  .. code-block:: c++

    template <typename...> struct TypeList;

    // The resulting type is TypeList<int, double, char>
    using MyTypeList = TypeList<__builtin_dedup_pack<int, double, int, char, double>...>;

  Currently, the use of ``__builtin_dedup_pack`` is limited to template arguments and base
  specifiers, it also must be used within a template context.

- ``__builtin_assume_dereferenceable`` now accepts non-constant size operands.

- Fixed a crash when the second argument to ``__builtin_assume_aligned`` was not constant (#GH161314)

- Introduce support for :doc:`allocation tokens <AllocToken>` to enable
  allocator-level heap organization strategies. A feature to instrument all
  allocation functions with a token ID can be enabled via the
  ``-fsanitize=alloc-token`` flag.

New Compiler Flags
------------------
- New option ``-fno-sanitize-debug-trap-reasons`` added to disable emitting trap reasons into the debug info when compiling with trapping UBSan (e.g. ``-fsanitize-trap=undefined``).
- New option ``-fsanitize-debug-trap-reasons=`` added to control emitting trap reasons into the debug info when compiling with trapping UBSan (e.g. ``-fsanitize-trap=undefined``).
- New options for enabling allocation token instrumentation: ``-fsanitize=alloc-token``, ``-falloc-token-max=``, ``-fsanitize-alloc-token-fast-abi``, ``-fsanitize-alloc-token-extended``.


Lanai Support
^^^^^^^^^^^^^^
- The option ``-mcmodel={small,medium,large}`` is supported again.

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------
- The `-gkey-instructions` compiler flag is now enabled by default when DWARF is emitted for plain C/C++ and optimizations are enabled. (#GH149509)
- The `-fconstexpr-steps` compiler flag now accepts value `0` to opt out of this limit. (#GH160440)

Removed Compiler Flags
-------------------------

Attribute Changes in Clang
--------------------------
- The definition of a function declaration with ``[[clang::cfi_unchecked_callee]]`` inherits this
  attribute, allowing the attribute to only be attached to the declaration. Prior, this would be
  treated as an error where the definition and declaration would have differing types.

- New format attributes ``gnu_printf``, ``gnu_scanf``, ``gnu_strftime`` and ``gnu_strfmon`` are added
  as aliases for ``printf``, ``scanf``, ``strftime`` and ``strfmon``. (#GH16219)

Improvements to Clang's diagnostics
-----------------------------------
- Added a separate diagnostic group ``-Wfunction-effect-redeclarations``, for the more pedantic
  diagnostics for function effects (``[[clang::nonblocking]]`` and ``[[clang::nonallocating]]``).
  Moved the warning for a missing (though implied) attribute on a redeclaration into this group.
  Added a new warning in this group for the case where the attribute is missing/implicit on
  an override of a virtual method.
- Implemented diagnostics when retrieving the tuple size for types where its specialization of `std::tuple_size`
  produces an invalid size (either negative or greater than the implementation limit). (#GH159563)
- Fixed fix-it hint for fold expressions. Clang now correctly places the suggested right
  parenthesis when diagnosing malformed fold expressions. (#GH151787)
- Added fix-it hint for when scoped enumerations require explicit conversions for binary operations. (#GH24265)
- Constant template parameters are now type checked in template definitions,
  including template template parameters.
- Fixed an issue where emitted format-signedness diagnostics were not associated with an appropriate
  diagnostic id. Besides being incorrect from an API standpoint, this was user visible, e.g.:
  "format specifies type 'unsigned int' but the argument has type 'int' [-Wformat]"
  "signedness of format specifier 'u' is incompatible with 'c' [-Wformat]"
  This was misleading, because even though -Wformat is required in order to emit the diagnostics,
  the warning flag the user needs to concerned with here is -Wformat-signedness, which is also
  required and is not enabled by default. With the change you'll now see:
  "format specifies type 'unsigned int' but the argument has type 'int', which differs in signedness [-Wformat-signedness]"
  "signedness of format specifier 'u' is incompatible with 'c' [-Wformat-signedness]"
  and the API-visible diagnostic id will be appropriate.
- Clang now produces better diagnostics for template template parameter matching
  involving 'auto' template parameters.
- Fixed false positives in ``-Waddress-of-packed-member`` diagnostics when
  potential misaligned members get processed before they can get discarded.
  (#GH144729)

- Clang now emits dignostic with correct message in case of assigning to const reference captured in lambda. (#GH105647)

- Fixed false positive in ``-Wmissing-noreturn`` diagnostic when it was requiring the usage of
  ``[[noreturn]]`` on lambdas before C++23 (#GH154493).

- Clang now diagnoses the use of ``#`` and ``##`` preprocessor tokens in
  attribute argument lists in C++ when ``-pedantic`` is enabled. The operators
  can be used in macro replacement lists with the usual preprocessor semantics,
  however, non-preprocessor use of tokens now triggers a pedantic warning in C++.
  Compilation in C mode is unchanged, and still permits these tokens to be used. (#GH147217)

- Clang now diagnoses misplaced array bounds on declarators for template
  specializations in th same way as it already did for other declarators.
  (#GH147333)

- A new warning ``-Walloc-size`` has been added to detect calls to functions
  decorated with the ``alloc_size`` attribute don't allocate enough space for
  the target pointer type.

- The :doc:`ThreadSafetyAnalysis` attributes ``ACQUIRED_BEFORE(...)`` and
  ``ACQUIRED_AFTER(...)`` have been moved to the stable feature set and no
  longer require ``-Wthread-safety-beta`` to be used.
- The :doc:`ThreadSafetyAnalysis` gains basic alias-analysis of capability
  pointers under ``-Wthread-safety-beta`` (still experimental), which reduces
  both false positives but also false negatives through more precise analysis.

- Clang now looks through parenthesis for ``-Wundefined-reinterpret-cast`` diagnostic.

- Fixed a bug where the source location was missing when diagnosing ill-formed
  placeholder constraints.

- The two-element, unary mask variant of ``__builtin_shufflevector`` is now
  properly being rejected when used at compile-time. It was not implemented
  and caused assertion failures before (#GH158471).

- Closed a loophole in the diagnosis of function pointer conversions changing
  extended function type information in C mode (#GH41465). Function conversions
  that were previously incorrectly accepted in case of other irrelevant
  conditions are now consistently diagnosed, identical to C++ mode.

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------
- Fix a crash when marco name is empty in ``#pragma push_macro("")`` or
  ``#pragma pop_macro("")``. (#GH149762).
- Fix a crash in variable length array (e.g. ``int a[*]``) function parameter type
  being used in ``_Countof`` expression. (#GH152826).
- ``-Wunreachable-code`` now diagnoses tautological or contradictory
  comparisons such as ``x != 0 || x != 1.0`` and ``x == 0 && x == 1.0`` on
  targets that treat ``_Float16``/``__fp16`` as native scalar types. Previously
  the warning was silently lost because the operands differed only by an implicit
  cast chain. (#GH149967).
- Fix crash in ``__builtin_function_start`` by checking for invalid
  first parameter. (#GH113323).
- Fixed a crash with incompatible pointer to integer conversions in designated
  initializers involving string literals. (#GH154046)
- Fix crash on CTAD for alias template. (#GH131342), (#GH131408)
- Clang now emits a frontend error when a function marked with the `flatten` attribute
  calls another function that requires target features not enabled in the caller. This
  prevents a fatal error in the backend.
- Fixed scope of typedefs present inside a template class. (#GH91451)
- Builtin elementwise operators now accept vector arguments that have different
  qualifiers on their elements. For example, vector of 4 ``const float`` values
  and vector of 4 ``float`` values. (#GH155405)
- Fixed inconsistent shadow warnings for lambda capture of structured bindings.
  Previously, ``[val = val]`` (regular parameter) produced no warnings with ``-Wshadow``
  while ``[a = a]`` (where ``a`` is from ``auto [a, b] = std::make_pair(1, 2)``)
  incorrectly produced warnings. Both cases now consistently show no warnings with
  ``-Wshadow`` and show uncaptured-local warnings with ``-Wshadow-all``. (#GH68605)
- Fixed a failed assertion with a negative limit parameter value inside of
  ``__has_embed``. (#GH157842)
- Fixed an assertion when an improper use of the ``malloc`` attribute targeting
  a function without arguments caused us to try to access a non-existent argument.
  (#GH159080)
- Fixed a failed assertion with empty filename arguments in ``__has_embed``. (#GH159898)
- Fixed a failed assertion with empty filename in ``#embed`` directive. (#GH162951)

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fix an ambiguous reference to the builtin `type_info` (available when using
  `-fms-compatibility`) with modules. (#GH38400)

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``[[nodiscard]]`` is now respected on Objective-C and Objective-C++ methods
  (#GH141504) and on types returned from indirect calls (#GH142453).
- Fixes some late parsed attributes, when applied to function definitions, not being parsed
  in function try blocks, and some situations where parsing of the function body
  is skipped, such as error recovery and code completion. (#GH153551)
- Using ``[[gnu::cleanup(some_func)]]`` where some_func is annotated with
  ``[[gnu::error("some error")]]`` now correctly triggers an error. (#GH146520)
- Fix a crash when the function name is empty in the `swift_name` attribute. (#GH157075)

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^
- Diagnose binding a reference to ``*nullptr`` during constant evaluation. (#GH48665)
- Suppress ``-Wdeprecated-declarations`` in implicitly generated functions. (#GH147293)
- Fix a crash when deleting a pointer to an incomplete array (#GH150359).
- Fixed a mismatched lambda scope bug when propagating up ``consteval`` within nested lambdas. (#GH145776)
- Disallow immediate escalation in destructors. (#GH109096)
- Fix an assertion failure when expression in assumption attribute
  (``[[assume(expr)]]``) creates temporary objects.
- Fix the dynamic_cast to final class optimization to correctly handle
  casts that are guaranteed to fail (#GH137518).
- Fix bug rejecting partial specialization of variable templates with auto NTTPs (#GH118190).
- Fix a crash if errors "member of anonymous [...] redeclares" and
  "intializing multiple members of union" coincide (#GH149985).
- Fix a crash when using ``explicit(bool)`` in pre-C++11 language modes. (#GH152729)
- Fix the parsing of variadic member functions when the ellipis immediately follows a default argument.(#GH153445)
- Fixed a bug that caused ``this`` captured by value in a lambda with a dependent explicit object parameter to not be
  instantiated properly. (#GH154054)
- Fixed a bug where our ``member-like constrained friend`` checking caused an incorrect analysis of lambda captures. (#GH156225)
- Fixed a crash when implicit conversions from initialize list to arrays of
  unknown bound during constant evaluation. (#GH151716)
- Support the dynamic_cast to final class optimization with pointer
  authentication enabled. (#GH152601)
- Fix the check for narrowing int-to-float conversions, so that they are detected in
  cases where converting the float back to an integer is undefined behaviour (#GH157067).
- Stop rejecting C++11-style attributes on the first argument of constructors in older
  standards. (#GH156809).
- Fix a crash when applying binary or ternary operators to two same function types with different spellings,
  where at least one of the function parameters has an attribute which affects
  the function type.
- Fix an assertion failure when a ``constexpr`` variable is only referenced through
  ``__builtin_addressof``, and related issues with builtin arguments. (#GH154034)
- Fix an assertion failure when taking the address on a non-type template parameter argument of
  object type. (#GH151531)
- Suppress ``-Wdouble-promotion`` when explicitly asked for with C++ list initialization (#GH33409).
- Fix the result of `__builtin_is_implicit_lifetime` for types with a user-provided constructor. (#GH160610)
- Correctly deduce return types in ``decltype`` expressions. (#GH160497) (#GH56652) (#GH116319) (#GH161196)
- Fixed a crash in the pre-C++23 warning for attributes before a lambda declarator (#GH161070).
- Fix a crash when attempting to deduce a deduction guide from a non deducible template template parameter. (#130604)
- Fix for clang incorrectly rejecting the default construction of a union with
  nontrivial member when another member has an initializer. (#GH81774)
- Diagnose unresolved overload sets in non-dependent compound requirements. (#GH51246) (#GH97753)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fix incorrect name qualifiers applied to alias CTAD. (#GH136624)
- Fixed ElaboratedTypes appearing within NestedNameSpecifier, which was not a
  legal representation. This is fixed because ElaboratedTypes don't exist anymore. (#GH43179) (#GH68670) (#GH92757)
- Fix unrecognized html tag causing undesirable comment lexing (#GH152944)
- Fix comment lexing of special command names (#GH152943)
- Use `extern` as a hint to continue parsing when recovering from a malformed declaration.

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^
- Fixed missing diagnostics of ``diagnose_if`` on templates involved in initialization. (#GH160776)

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
- More SSE, AVX and AVX512 intrinsics, including initializers and general
  arithmetic can now be used in C++ constant expressions.
- Some SSE, AVX and AVX512 intrinsics have been converted to wrap
  generic __builtin intrinsics.
- NOTE: Please avoid use of the __builtin_ia32_* intrinsics - these are not
  guaranteed to exist in future releases, or match behaviour with previous
  releases of clang or other compilers.
- Remove `m[no-]avx10.x-[256,512]` and `m[no-]evex512` options from Clang
  driver.
- Remove `[no-]evex512` feature request from intrinsics and builtins.
- Change features `avx10.x-[256,512]` to `avx10.x`.

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^
- Enable linker relaxation by default for loongarch64.

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

Support calling `consteval` function between different target.

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
- Remove the ``dependentTemplateSpecializationType`` matcher, as the
  corresponding AST node was removed. This matcher was never very useful, since
  there was no way to match on its template name.
- Add a boolean member ``IgnoreSystemHeaders`` to ``MatchFinderOptions``. This
  allows it to ignore nodes in system headers when traversing the AST.

- ``hasConditionVariableStatement`` now supports ``for`` loop, ``while`` loop
  and ``switch`` statements.

clang-format
------------
- Add ``SpaceInEmptyBraces`` option and set it to ``Always`` for WebKit style.
- Add ``NumericLiteralCase`` option for enforcing character case in numeric
  literals.
- Add ``Leave`` suboption to ``IndentPPDirectives``.
- Add ``AllowBreakBeforeQtProperty`` option.

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
- Fixed a crash when compiling ``__real__`` or ``__imag__`` unary operator on scalar value with type promotion. (#GH160583)

Improvements
^^^^^^^^^^^^

Moved checkers
^^^^^^^^^^^^^^

.. _release-notes-sanitizers:

Sanitizers
----------
- Improved documentation for legacy ``no_sanitize`` attributes.

Python Binding Changes
----------------------
- Exposed ``clang_getCursorLanguage`` via ``Cursor.language``.
- Add all missing ``CursorKind``s, ``TypeKind``s and
  ``ExceptionSpecificationKind``s from ``Index.h``

OpenMP Support
--------------
- Added parsing and semantic analysis support for the ``need_device_addr``
  modifier in the ``adjust_args`` clause.
- Allow array length to be omitted in array section subscript expression.
- Fixed non-contiguous strided update in the ``omp target update`` directive with the ``from`` clause.
- Properly handle array section/assumed-size array privatization in C/C++.
- Added support to handle new syntax of the ``uses_allocators`` clause.
- Added support for ``variable-category`` modifier in ``default clause``.
- Added support for ``defaultmap`` directive implicit-behavior ``storage``.
- Added support for ``defaultmap`` directive implicit-behavior ``private``.
- Added parsing and semantic analysis support for ``groupprivate`` directive.
- Added support for 'omp fuse' directive.
- Updated parsing and semantic analysis support for ``nowait`` clause to accept
  optional argument in OpenMP >= 60.

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
