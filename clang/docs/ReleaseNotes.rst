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

C/C++ Language Potentially Breaking Changes
-------------------------------------------

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
- Members of anonymous unions/structs are now injected as ``IndirectFieldDecl``
  into the enclosing record even if their names conflict with other names in the
  scope. These ``IndirectFieldDecl`` are marked invalid.

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

- ``__builtin_elementwise_abs`` can now be used in constant expression.

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

- A vector of booleans is now a valid condition for the ternary ``?:`` operator.
  This binds to a simple vector select operation.

- Added ``__builtin_masked_load`` and ``__builtin_masked_store`` for conditional
  memory loads from vectors. Binds to the LLVM intrinsic of the same name.

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


New Compiler Flags
------------------
- New option ``-fno-sanitize-annotate-debug-info-traps`` added to disable emitting trap reasons into the debug info when compiling with trapping UBSan (e.g. ``-fsanitize-trap=undefined``).

Lanai Support
^^^^^^^^^^^^^^
- The option ``-mcmodel={small,medium,large}`` is supported again.

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

- Improve the diagnostics for deleted default constructor errors for C++ class
  initializer lists that don't explicitly list a class member and thus attempt
  to implicitly default construct that member.
- The ``-Wunique-object-duplication`` warning has been added to warn about objects
  which are supposed to only exist once per program, but may get duplicated when
  built into a shared library.
- Fixed a bug where Clang's Analysis did not correctly model the destructor behavior of ``union`` members (#GH119415).
- A statement attribute applied to a ``case`` label no longer suppresses
  'bypassing variable initialization' diagnostics (#84072).
- The ``-Wunsafe-buffer-usage`` warning has been updated to warn
  about unsafe libc function calls.  Those new warnings are emitted
  under the subgroup ``-Wunsafe-buffer-usage-in-libc-call``.
- Diagnostics on chained comparisons (``a < b < c``) are now an error by default. This can be disabled with
  ``-Wno-error=parentheses``.
- Similarly, fold expressions over a comparison operator are now an error by default.
- Clang now better preserves the sugared types of pointers to member.
- Clang now better preserves the presence of the template keyword with dependent
  prefixes.
- Clang now in more cases avoids printing 'type-parameter-X-X' instead of the name of
  the template parameter.
- Clang now respects the current language mode when printing expressions in
  diagnostics. This fixes a bunch of `bool` being printed as `_Bool`, and also
  a bunch of HLSL types being printed as their C++ equivalents.
- Clang now consistently quotes expressions in diagnostics.
- When printing types for diagnostics, clang now doesn't suppress the scopes of
  template arguments contained within nested names.
- The ``-Wshift-bool`` warning has been added to warn about shifting a boolean. (#GH28334)
- Fixed diagnostics adding a trailing ``::`` when printing some source code
  constructs, like base classes.
- The :doc:`ThreadSafetyAnalysis` now supports ``-Wthread-safety-pointer``,
  which enables warning on passing or returning pointers to guarded variables
  as function arguments or return value respectively. Note that
  :doc:`ThreadSafetyAnalysis` still does not perform alias analysis. The
  feature will be default-enabled with ``-Wthread-safety`` in a future release.
- The :doc:`ThreadSafetyAnalysis` now supports reentrant capabilities.
- Clang will now do a better job producing common nested names, when producing
  common types for ternary operator, template argument deduction and multiple return auto deduction.
- The ``-Wsign-compare`` warning now treats expressions with bitwise not(~) and minus(-) as signed integers
  except for the case where the operand is an unsigned integer
  and throws warning if they are compared with unsigned integers (##18878).
- The ``-Wunnecessary-virtual-specifier`` warning (included in ``-Wextra``) has
  been added to warn about methods which are marked as virtual inside a
  ``final`` class, and hence can never be overridden.

- Improve the diagnostics for chained comparisons to report actual expressions and operators (#GH129069).

- Improve the diagnostics for shadows template parameter to report correct location (#GH129060).

- Improve the ``-Wundefined-func-template`` warning when a function template is not instantiated due to being unreachable in modules.

- When diagnosing an unused return value of a type declared ``[[nodiscard]]``, the type
  itself is now included in the diagnostic.

- Clang will now prefer the ``[[nodiscard]]`` declaration on function declarations over ``[[nodiscard]]``
  declaration on the return type of a function. Previously, when both have a ``[[nodiscard]]`` declaration attached,
  the one on the return type would be preferred. This may affect the generated warning message:

  .. code-block:: c++

    struct [[nodiscard("Reason 1")]] S {};
    [[nodiscard("Reason 2")]] S getS();
    void use()
    {
      getS(); // Now diagnoses "Reason 2", previously diagnoses "Reason 1"
    }

- Fixed an assertion when referencing an out-of-bounds parameter via a function
  attribute whose argument list refers to parameters by index and the function
  is variadic. e.g.,

  .. code-block:: c

    __attribute__ ((__format_arg__(2))) void test (int i, ...) { }

  Fixes #GH61635

- Split diagnosing base class qualifiers from the ``-Wignored-Qualifiers`` diagnostic group into a new ``-Wignored-base-class-qualifiers`` diagnostic group (which is grouped under ``-Wignored-qualifiers``). Fixes #GH131935.

- ``-Wc++98-compat`` no longer diagnoses use of ``__auto_type`` or
  ``decltype(auto)`` as though it was the extension for ``auto``. (#GH47900)
- Clang now issues a warning for missing return in ``main`` in C89 mode. (#GH21650)

- Now correctly diagnose a tentative definition of an array with static
  storage duration in pedantic mode in C. (#GH50661)
- No longer diagnosing idiomatic function pointer casts on Windows under
  ``-Wcast-function-type-mismatch`` (which is enabled by ``-Wextra``). Clang
  would previously warn on this construct, but will no longer do so on Windows:

  .. code-block:: c

    typedef void (WINAPI *PGNSI)(LPSYSTEM_INFO);
    HMODULE Lib = LoadLibrary("kernel32");
    PGNSI FnPtr = (PGNSI)GetProcAddress(Lib, "GetNativeSystemInfo");


- An error is now emitted when a ``musttail`` call is made to a function marked with the ``not_tail_called`` attribute. (#GH133509).

- ``-Whigher-precision-for-complex-divison`` warns when:

  -	The divisor is complex.
  -	When the complex division happens in a higher precision type due to arithmetic promotion.
  -	When using the divide and assign operator (``/=``).

  Fixes #GH131127

- ``-Wuninitialized`` now diagnoses when a class does not declare any
  constructors to initialize their non-modifiable members. The diagnostic is
  not new; being controlled via a warning group is what's new. Fixes #GH41104

- Analysis-based diagnostics (like ``-Wconsumed`` or ``-Wunreachable-code``)
  can now be correctly controlled by ``#pragma clang diagnostic``. #GH42199

- Improved Clang's error recovery for invalid function calls.

- Improved bit-field diagnostics to consider the type specified by the
  ``preferred_type`` attribute. These diagnostics are controlled by the flags
  ``-Wpreferred-type-bitfield-enum-conversion`` and
  ``-Wpreferred-type-bitfield-width``. These warnings are on by default as they
  they're only triggered if the authors are already making the choice to use
  ``preferred_type`` attribute.

- ``-Winitializer-overrides`` and ``-Wreorder-init-list`` are now grouped under
  the ``-Wc99-designator`` diagnostic group, as they also are about the
  behavior of the C99 feature as it was introduced into C++20. Fixes #GH47037
- ``-Wreserved-identifier`` now fires on reserved parameter names in a function
  declaration which is not a definition.
- Clang now prints the namespace for an attribute, if any,
  when emitting an unknown attribute diagnostic.

- ``-Wvolatile`` now warns about volatile-qualified class return types
  as well as volatile-qualified scalar return types. Fixes #GH133380

- Several compatibility diagnostics that were incorrectly being grouped under
  ``-Wpre-c++20-compat`` are now part of ``-Wc++20-compat``. (#GH138775)

- Improved the ``-Wtautological-overlap-compare`` diagnostics to warn about overlapping and non-overlapping ranges involving character literals and floating-point literals.
  The warning message for non-overlapping cases has also been improved (#GH13473).

- Fixed a duplicate diagnostic when performing typo correction on function template
  calls with explicit template arguments. (#GH139226)

- Explanatory note is printed when ``assert`` fails during evaluation of a
  constant expression. Prior to this, the error inaccurately implied that assert
  could not be used at all in a constant expression (#GH130458)

- A new off-by-default warning ``-Wms-bitfield-padding`` has been added to alert to cases where bit-field
  packing may differ under the MS struct ABI (#GH117428).

- ``-Watomic-access`` no longer fires on unreachable code. e.g.,

  .. code-block:: c

    _Atomic struct S { int a; } s;
    void func(void) {
      if (0)
        s.a = 12; // Previously diagnosed with -Watomic-access, now silenced
      s.a = 12; // Still diagnosed with -Watomic-access
      return;
      s.a = 12; // Previously diagnosed, now silenced
    }


- A new ``-Wcharacter-conversion`` warns where comparing or implicitly converting
  between different Unicode character types (``char8_t``, ``char16_t``, ``char32_t``).
  This warning only triggers in C++ as these types are aliases in C. (#GH138526)

- Fixed a crash when checking a ``__thread``-specified variable declaration
  with a dependent type in C++. (#GH140509)

- Clang now suggests corrections for unknown attribute names.

- ``-Wswitch`` will now diagnose unhandled enumerators in switches also when
  the enumerator is deprecated. Warnings about using deprecated enumerators in
  switch cases have moved behind a new ``-Wdeprecated-declarations-switch-case``
  flag.

  For example:

  .. code-block:: c

    enum E {
      Red,
      Green,
      Blue [[deprecated]]
    };
    void example(enum E e) {
      switch (e) {
      case Red:   // stuff...
      case Green: // stuff...
      }
    }

  will result in a warning about ``Blue`` not being handled in the switch.

  The warning can be fixed either by adding a ``default:``, or by adding
  ``case Blue:``. Since the enumerator is deprecated, the latter approach will
  trigger a ``'Blue' is deprecated`` warning, which can be turned off with
  ``-Wno-deprecated-declarations-switch-case``.

- Split diagnosis of implicit integer comparison on negation to a new
  diagnostic group ``-Wimplicit-int-comparison-on-negation``, grouped under
  ``-Wimplicit-int-conversion``, so user can turn it off independently.

- Improved the FixIts for unused lambda captures.

- Delayed typo correction was removed from the compiler; immediate typo
  correction behavior remains the same. Delayed typo correction facilities were
  fragile and unmaintained, and the removal closed the following issues:
  #GH142457, #GH139913, #GH138850, #GH137867, #GH137860, #GH107840, #GH93308,
  #GH69470, #GH59391, #GH58172, #GH46215, #GH45915, #GH45891, #GH44490,
  #GH36703, #GH32903, #GH23312, #GH69874.

- Clang no longer emits a spurious -Wdangling-gsl warning in C++23 when
  iterating over an element of a temporary container in a range-based
  for loop.(#GH109793, #GH145164)

- Fixed false positives in ``-Wformat-truncation`` and ``-Wformat-overflow``
  diagnostics when floating-point numbers had both width field and plus or space
  prefix specified. (#GH143951)

- A warning is now emitted when ``main`` is attached to a named module,
  which can be turned off with ``-Wno-main-attached-to-named-module``. (#GH146247)

- Clang now avoids issuing `-Wreturn-type` warnings in some cases where
  the final statement of a non-void function is a `throw` expression, or
  a call to a function that is trivially known to always throw (i.e., its
  body consists solely of a `throw` statement). This avoids certain
  false positives in exception-heavy code, though only simple patterns
  are currently recognized.

- Clang now accepts ``@tparam`` comments on variable template partial
  specializations. (#GH144775)

- Fixed a bug that caused diagnostic line wrapping to not function correctly on
  some systems. (#GH139499)

- Clang now tries to avoid printing file paths that contain ``..``, instead preferring
  the canonical file path if it ends up being shorter.

- Improve the diagnostics for placement new expression when const-qualified
  object was passed as the storage argument. (#GH143708)

- Clang now does not issue a warning about returning from a function declared with
  the ``[[noreturn]]`` attribute when the function body is ended with a call via
  pointer, provided it can be proven that the pointer only points to
  ``[[noreturn]]`` functions.

- Added a separate diagnostic group ``-Wfunction-effect-redeclarations``, for the more pedantic
  diagnostics for function effects (``[[clang::nonblocking]]`` and ``[[clang::nonallocating]]``).
  Moved the warning for a missing (though implied) attribute on a redeclaration into this group.
  Added a new warning in this group for the case where the attribute is missing/implicit on
  an override of a virtual method.
- Fixed fix-it hint for fold expressions. Clang now correctly places the suggested right
  parenthesis when diagnosing malformed fold expressions. (#GH151787)
- Added fix-it hint for when scoped enumerations require explicit conversions for binary operations. (#GH24265)

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
  
- Fixed false positives in ``-Waddress-of-packed-member`` diagnostics when
  potential misaligned members get processed before they can get discarded.
  (#GH144729)

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
- `-Wunreachable-code`` now diagnoses tautological or contradictory
  comparisons such as ``x != 0 || x != 1.0`` and ``x == 0 && x == 1.0`` on
  targets that treat ``_Float16``/``__fp16`` as native scalar types. Previously
  the warning was silently lost because the operands differed only by an implicit
  cast chain. (#GH149967).
- Fixed a crash with incompatible pointer to integer conversions in designated
  initializers involving string literals. (#GH154046)
- Clang now emits a frontend error when a function marked with the `flatten` attribute
  calls another function that requires target features not enabled in the caller. This
  prevents a fatal error in the backend.
- Fixed scope of typedefs present inside a template class. (#GH91451)

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

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^
- Diagnose binding a reference to ``*nullptr`` during constant evaluation. (#GH48665)
- Suppress ``-Wdeprecated-declarations`` in implicitly generated functions. (#GH147293)
- Fix a crash when deleting a pointer to an incomplete array (#GH150359).
- Fixed a mismatched lambda scope bug when propagating up ``consteval`` within nested lambdas. (#GH145776)
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

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fix incorrect name qualifiers applied to alias CTAD. (#GH136624)
- Fixed ElaboratedTypes appearing within NestedNameSpecifier, which was not a
  legal representation. This is fixed because ElaboratedTypes don't exist anymore. (#GH43179) (#GH68670) (#GH92757)
- Fix unrecognized html tag causing undesirable comment lexing (#GH152944)
- Fix comment lexing of special command names (#GH152943)

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

- Bump the default code object version to 6. ROCm 6.3 is required to run any program compiled with COV6.
- Introduced a new target specific builtin ``__builtin_amdgcn_processor_is``,
  a late / deferred query for the current target processor
- Introduced a new target specific builtin ``__builtin_amdgcn_is_invocable``,
  which enables fine-grained, per-builtin, feature availability

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

- Add a boolean member ``IgnoreSystemHeaders`` to ``MatchFinderOptions``. This
  allows it to ignore nodes in system headers when traversing the AST.

- ``hasConditionVariableStatement`` now supports ``for`` loop, ``while`` loop
  and ``switch`` statements.

clang-format
------------
- Add ``SpaceInEmptyBraces`` option and set it to ``Always`` for WebKit style.

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
