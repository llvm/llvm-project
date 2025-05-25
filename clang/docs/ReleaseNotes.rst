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

- The Objective-C ARC migrator (ARCMigrate) has been removed.
- Fix missing diagnostics for uses of declarations when performing typename access,
  such as when performing member access on a '[[deprecated]]' type alias.
  (#GH58547)
- For ARM targets when compiling assembly files, the features included in the selected CPU
  or Architecture's FPU are included. If you wish not to use a specific feature,
  the relevant ``+no`` option will need to be amended to the command line option.

C/C++ Language Potentially Breaking Changes
-------------------------------------------

- New LLVM optimizations have been implemented that optimize pointer arithmetic on
  null pointers more aggressively.  As part of this, clang has implemented a special
  case for old-style offsetof idioms like ``((int)(&(((struct S *)0)->field)))``, to
  ensure they are not caught by these optimizations.  It is also possible to use
  ``-fwrapv-pointer`` or   ``-fno-delete-null-pointer-checks`` to make pointer arithmetic
  on null pointers well-defined. (#GH130734, #GH130742, #GH130952)

C++ Specific Potentially Breaking Changes
-----------------------------------------

- The type trait builtin ``__is_referenceable`` has been removed, since it has
  very few users and all the type traits that could benefit from it in the
  standard library already have their own bespoke builtins.
- A workaround for libstdc++4.7 has been removed. Note that 4.8.3 remains the oldest
  supported libstdc++ version.

ABI Changes in This Version
---------------------------

- Return larger CXX records in memory instead of using AVX registers. Code compiled with older clang will be incompatible with newer version of the clang unless -fclang-abi-compat=20 is provided. (#GH120670)

AST Dumping Potentially Breaking Changes
----------------------------------------

- Added support for dumping template arguments of structural value kinds.

Clang Frontend Potentially Breaking Changes
-------------------------------------------

- The ``-Wglobal-constructors`` flag now applies to ``[[gnu::constructor]]`` and
  ``[[gnu::destructor]]`` attributes.

Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------
- ``Cursor.from_location`` now returns ``None`` instead of a null cursor.
  This eliminates the last known source of null cursors.
- Almost all ``Cursor`` methods now assert that they are called on non-null cursors.
  Most of the time null cursors were mapped to ``None``,
  so no widespread breakages are expected.

What's New in Clang |release|?
==============================

C++ Language Changes
--------------------

- Added a :ref:`__builtin_structured_binding_size <builtin_structured_binding_size-doc>` (T)
  builtin that returns the number of structured bindings that would be produced by destructuring ``T``.

- Similarly to GCC, Clang now supports constant expressions in
  the strings of a GNU ``asm`` statement.

  .. code-block:: c++

    int foo() {
      asm((std::string_view("nop")) ::: (std::string_view("memory")));
    }

- Clang now implements the changes to overload resolution proposed by section 1 and 2 of
  `P3606 <https://wg21.link/P3606R0>`_. If a non-template candidate exists in an overload set that is
  a perfect match (all conversion sequences are identity conversions) template candidates are not instantiated.
  Diagnostics that would have resulted from the instantiation of these template candidates are no longer
  produced. This aligns Clang closer to the behavior of GCC, and fixes (#GH62096), (#GH74581), and (#GH74581).

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P1061R10 Structured Bindings can introduce a Pack <https://wg21.link/P1061R10>`_.
- Implemented `P2786R13 Trivial Relocatability <https://wg21.link/P2786R13>`_.


- Implemented `P0963R3 Structured binding declaration as a condition <https://wg21.link/P0963R3>`_.

- Implemented `P2719R4 Type-aware allocation and deallocation functions <https://wg21.link/P2719>`_.

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Fixed a crash with a defaulted spaceship (``<=>``) operator when the class
  contains a member declaration of vector type. Vector types cannot yet be
  compared directly, so this causes the operator to be deleted. (#GH137452)

C++17 Feature Support
^^^^^^^^^^^^^^^^^^^^^

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The flag `-frelaxed-template-template-args`
  and its negation have been removed, having been deprecated since the previous
  two releases. The improvements to template template parameter matching implemented
  in the previous release, as described in P3310 and P3579, made this flag unnecessary.

- Implemented `CWG2918 Consideration of constraints for address of overloaded `
  `function <https://cplusplus.github.io/CWG/issues/2918.html>`_

- Bumped the ``__cpp_constexpr`` feature-test macro to ``202002L`` in C++20 mode as indicated in
  `P2493R0 <https://wg21.link/P2493R0>`_.

- Implemented `CWG3005 Function parameters should never be name-independent <https://wg21.link/CWG3005>`_.

C Language Changes
------------------

- Clang now allows an ``inline`` specifier on a typedef declaration of a
  function type in Microsoft compatibility mode. #GH124869
- Clang now allows ``restrict`` qualifier for array types with pointer elements (#GH92847).
- Clang now diagnoses ``const``-qualified object definitions without an
  initializer. If the object is a variable or field which is zero-initialized,
  it will be diagnosed under the new warning ``-Wdefault-const-init-var`` or
  ``-Wdefault-const-init-field``, respectively. Similarly, if the variable or
  field is not zero-initialized, it will be diagnosed under the new diagnostic
  ``-Wdefault-const-init-var-unsafe`` or ``-Wdefault-const-init-field-unsafe``,
  respectively. The unsafe diagnostic variants are grouped under a new
  diagnostic ``-Wdefault-const-init-unsafe``, which itself is grouped under the
  new diagnostic ``-Wdefault-const-init``. Finally, ``-Wdefault-const-init`` is
  grouped under ``-Wc++-compat`` because these constructs are not compatible
  with C++. #GH19297
- Added ``-Wimplicit-void-ptr-cast``, grouped under ``-Wc++-compat``, which
  diagnoses implicit conversion from ``void *`` to another pointer type as
  being incompatible with C++. (#GH17792)
- Added ``-Wc++-keyword``, grouped under ``-Wc++-compat``, which diagnoses when
  a C++ keyword is used as an identifier in C. (#GH21898)
- Added ``-Wc++-hidden-decl``, grouped under ``-Wc++-compat``, which diagnoses
  use of tag types which are visible in C but not visible in C++ due to scoping
  rules. e.g.,

  .. code-block:: c

    struct S {
      struct T {
        int x;
      } t;
    };
    struct T t; // Invalid C++, valid C, now diagnosed
- Added ``-Wimplicit-int-enum-cast``, grouped under ``-Wc++-compat``, which
  diagnoses implicit conversion from integer types to an enumeration type in C,
  which is not compatible with C++. #GH37027
- Split "implicit conversion from enum type to different enum type" diagnostic
  from ``-Wenum-conversion`` into its own diagnostic group,
  ``-Wimplicit-enum-enum-cast``, which is grouped under both
  ``-Wenum-conversion`` and ``-Wimplicit-int-enum-cast``. This conversion is an
  int-to-enum conversion because the enumeration on the right-hand side is
  promoted to ``int`` before the assignment.
- Added ``-Wtentative-definition-compat``, grouped under ``-Wc++-compat``,
  which diagnoses tentative definitions in C with multiple declarations as
  being incompatible with C++. e.g.,

  .. code-block:: c

    // File scope
    int i;
    int i; // Vaild C, invalid C++, now diagnosed
- Added ``-Wunterminated-string-initialization``, grouped under ``-Wextra``,
  which diagnoses an initialization from a string literal where only the null
  terminator cannot be stored. e.g.,

  .. code-block:: c


    char buf1[3] = "foo"; // -Wunterminated-string-initialization
    char buf2[3] = "flarp"; // -Wexcess-initializers

  This diagnostic can be suppressed by adding the new ``nonstring`` attribute
  to the field or variable being initialized. #GH137705
- Added ``-Wc++-unterminated-string-initialization``, grouped under
  ``-Wc++-compat``, which also diagnoses the same cases as
  ``-Wunterminated-string-initialization``. However, this diagnostic is not
  silenced by the ``nonstring`` attribute as these initializations are always
  incompatible with C++.
- Added ``-Wjump-misses-init``, which is off by default and grouped under
  ``-Wc++-compat``. It diagnoses when a jump (``goto`` to its label, ``switch``
  to its ``case``) will bypass the initialization of a local variable, which is
  invalid in C++.
- Added the existing ``-Wduplicate-decl-specifier`` diagnostic, which is on by
  default, to ``-Wc++-compat`` because duplicated declaration specifiers are
  not valid in C++.

C2y Feature Support
^^^^^^^^^^^^^^^^^^^
- Implement `WG14 N3409 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3409.pdf>`_
  which removes UB around use of ``void`` expressions. In practice, this means
  that ``_Generic`` selection associations may now have ``void`` type, but it
  also removes UB with code like ``(void)(void)1;``.
- Implemented `WG14 N3411 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3411.pdf>`_
  which allows a source file to not end with a newline character. Note,
  ``-pedantic`` will no longer diagnose this in either C or C++ modes. This
  feature was adopted as applying to obsolete versions of C in WG14 and as a
  defect report in WG21 (CWG787).
- Implemented `WG14 N3353 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3353.htm>`_
  which adds the new ``0o`` and ``0O`` ocal literal prefixes and deprecates
  octal literals other than ``0`` which do not start with the new prefix. This
  feature is exposed in earlier language modes and in C++ as an extension. The
  paper also introduced octal and hexadecimal delimited escape sequences (e.g.,
  ``"\x{12}\o{12}"``) which are also supported as an extension in older C
  language modes.
- Implemented `WG14 N3369 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3369.pdf>`_
  which introduces the ``_Lengthof`` operator, and `WG14 N3469 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3469.htm>`_
  which renamed ``_Lengthof`` to ``_Countof``. This feature is implemented as
  a conforming extension in earlier C language modes, but not in C++ language
  modes (``std::extent`` and ``std::size`` already provide the same
  functionality but with more granularity). The feature can be tested via
  ``__has_feature(c_countof)`` or ``__has_extension(c_countof)``.

C23 Feature Support
^^^^^^^^^^^^^^^^^^^
- Clang now accepts ``-std=iso9899:2024`` as an alias for C23.
- Added ``__builtin_c23_va_start()`` for compatibility with GCC and to enable
  better diagnostic behavior for the ``va_start()`` macro in C23 and later.
  This also updates the definition of ``va_start()`` in ``<stdarg.h>`` to use
  the new builtin. Fixes #GH124031.
- Implemented `WG14 N2819 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2819.pdf>`_
  which clarified that a compound literal used within a function prototype is
  treated as if the compound literal were within the body rather than at file
  scope.
- Fixed a bug where you could not cast a null pointer constant to type
  ``nullptr_t``. Fixes #GH133644.
- Implemented `WG14 N3037 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3037.pdf>`_
  which allows tag types to be redefined within the same translation unit so
  long as both definitions are structurally equivalent (same tag types, same
  tag names, same tag members, etc). As a result of this paper, ``-Wvisibility``
  is no longer diagnosed in C23 if the parameter is a complete tag type (it
  does still fire when the parameter is an incomplete tag type as that cannot
  be completed).
- Fixed a failed assertion with an invalid parameter to the ``#embed``
  directive. Fixes #GH126940.
- Fixed a crash when a declaration of a ``constexpr`` variable with an invalid
  type. Fixes #GH140887

C11 Feature Support
^^^^^^^^^^^^^^^^^^^
- Implemented `WG14 N1285 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1285.htm>`_
  which introduces the notion of objects with a temporary lifetime. When an
  expression resulting in an rvalue with structure or union type and that type
  contains a member of array type, the expression result is an automatic storage
  duration object with temporary lifetime which begins when the expression is
  evaluated and ends at the evaluation of the containing full expression. This
  functionality is also implemented for earlier C language modes because the
  C99 semantics will never be implemented (it would require dynamic allocations
  of memory which leaks, which users would not appreciate).

Non-comprehensive list of changes in this release
-------------------------------------------------

- Support parsing the `cc` operand modifier and alias it to the `c` modifier (#GH127719).
- Added `__builtin_elementwise_exp10`.
- For AMDPGU targets, added `__builtin_v_cvt_off_f32_i4` that maps to the `v_cvt_off_f32_i4` instruction.
- Added `__builtin_elementwise_minnum` and `__builtin_elementwise_maxnum`.
- No longer crashing on invalid Objective-C categories and extensions when
  dumping the AST as JSON. (#GH137320)
- Clang itself now uses split stacks instead of threads for allocating more
  stack space when running on Apple AArch64 based platforms. This means that
  stack traces of Clang from debuggers, crashes, and profilers may look
  different than before.
- Fixed a crash when a VLA with an invalid size expression was used within a
  ``sizeof`` or ``typeof`` expression. (#GH138444)

New Compiler Flags
------------------

- New option ``-Wundef-true`` added and enabled by default to warn when `true` is used in the C preprocessor without being defined before C23.

- New option ``-fprofile-continuous`` added to enable continuous profile syncing to file (#GH124353, `docs <https://clang.llvm.org/docs/UsersManual.html#cmdoption-fprofile-continuous>`_).
  The feature has `existed <https://clang.llvm.org/docs/SourceBasedCodeCoverage.html#running-the-instrumented-program>`_)
  for a while and this is just a user facing option.

- New option ``-ftime-report-json`` added which outputs the same timing data as ``-ftime-report`` but formatted as JSON.

- New option ``-Wnrvo`` added and disabled by default to warn about missed NRVO opportunities.

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

- The ARM AArch32 ``-mtp`` option accepts and defaults to ``auto``, a value of ``auto`` uses the best available method of providing the frame pointer supported by the hardware. This matches
  the behavior of ``-mtp`` in gcc. This changes the default behavior for ARM targets that provide the ``TPIDRURO`` register as this will be used instead of a call to the ``__aeabi_read_tp``.
  Programs that use ``__aeabi_read_tp`` but do not use the ``TPIDRURO`` register must use ``-mtp=soft``. Fixes #123864

- The compiler flag `-fbracket-depth` default value is increased from 256 to 2048. (#GH94728)

- `-Wpadded` option implemented for the `x86_64-windows-msvc` target. Fixes #61702

- The ``-mexecute-only`` and ``-mpure-code`` flags are now accepted for AArch64 targets. (#GH125688)

- The ``-fchar8_t`` flag is no longer considered in non-C++ languages modes. (#GH55373)

Removed Compiler Flags
-------------------------

Attribute Changes in Clang
--------------------------
Adding [[clang::unsafe_buffer_usage]] attribute to a method definition now turns off all -Wunsafe-buffer-usage
related warnings within the method body.

- The ``no_sanitize`` attribute now accepts both ``gnu`` and ``clang`` names.
- The ``ext_vector_type(n)`` attribute can now be used as a generic type attribute.
- Clang now diagnoses use of declaration attributes on void parameters. (#GH108819)
- Clang now allows ``__attribute__((model("small")))`` and
  ``__attribute__((model("large")))`` on non-TLS globals in x86-64 compilations.
  This forces the global to be considered small or large in regards to the
  x86-64 code model, regardless of the code model specified for the compilation.
- Clang now emits a warning ``-Wreserved-init-priority`` instead of a hard error
  when ``__attribute__((init_priority(n)))`` is used with values of n in the
  reserved range [0, 100]. The warning will be treated as an error by default.

- There is a new ``format_matches`` attribute to complement the existing
  ``format`` attribute. ``format_matches`` allows the compiler to verify that
  a format string argument is equivalent to a reference format string: it is
  useful when a function accepts a format string without its accompanying
  arguments to format. For instance:

  .. code-block:: c

    static int status_code;
    static const char *status_string;

    void print_status(const char *fmt) {
      fprintf(stderr, fmt, status_code, status_string);
      // ^ warning: format string is not a string literal [-Wformat-nonliteral]
    }

    void stuff(void) {
      print_status("%s (%#08x)\n");
      // order of %s and %x is swapped but there is no diagnostic
    }

  Before the introducion of ``format_matches``, this code cannot be verified
  at compile-time. ``format_matches`` plugs that hole:

  .. code-block:: c

    __attribute__((format_matches(printf, 1, "%x %s")))
    void print_status(const char *fmt) {
      fprintf(stderr, fmt, status_code, status_string);
      // ^ `fmt` verified as if it was "%x %s" here; no longer triggers
      //   -Wformat-nonliteral, would warn if arguments did not match "%x %s"
    }

    void stuff(void) {
      print_status("%s (%#08x)\n");
      // warning: format specifier 's' is incompatible with 'x'
      // warning: format specifier 'x' is incompatible with 's'
    }

  Like with ``format``, the first argument is the format string flavor and the
  second argument is the index of the format string parameter.
  ``format_matches`` accepts an example valid format string as its third
  argument. For more information, see the Clang attributes documentation.

- Introduced a new statement attribute ``[[clang::atomic]]`` that enables
  fine-grained control over atomic code generation on a per-statement basis.
  Supported options include ``[no_]remote_memory``,
  ``[no_]fine_grained_memory``, and ``[no_]ignore_denormal_mode``. These are
  particularly relevant for AMDGPU targets, where they map to corresponding IR
  metadata.

- Clang now disallows the use of attributes applied before an
  ``extern template`` declaration (#GH79893).

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
  switch cases have moved behind a new ``-Wdeprecated-switch-case`` flag.

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
  ``-Wno-deprecated-switch-case``.

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------

- Clang now outputs correct values when #embed data contains bytes with negative
  signed char values (#GH102798).
- Fixed a crash when merging named enumerations in modules (#GH114240).
- Fixed rejects-valid problem when #embed appears in std::initializer_list or
  when it can affect template argument deduction (#GH122306).
- Fix crash on code completion of function calls involving partial order of function templates
  (#GH125500).
- Fixed clang crash when #embed data does not fit into an array
  (#GH128987).
- Non-local variable and non-variable declarations in the first clause of a ``for`` loop in C are no longer incorrectly
  considered an error in C23 mode and are allowed as an extension in earlier language modes.

- Remove the ``static`` specifier for the value of ``_FUNCTION_`` for static functions, in MSVC compatibility mode.
- Fixed a modules crash where exception specifications were not propagated properly (#GH121245, relanded in #GH129982)
- Fixed a problematic case with recursive deserialization within ``FinishedDeserializing()`` where
  ``PassInterestingDeclsToConsumer()`` was called before the declarations were safe to be passed. (#GH129982)
- Fixed a modules crash where an explicit Constructor was deserialized. (#GH132794)
- Defining an integer literal suffix (e.g., ``LL``) before including
  ``<stdint.h>`` in a freestanding build no longer causes invalid token pasting
  when using the ``INTn_C`` macros. (#GH85995)
- Fixed an assertion failure in the expansion of builtin macros like ``__has_embed()`` with line breaks before the
  closing paren. (#GH133574)
- Fixed a crash in error recovery for expressions resolving to templates. (#GH135621)
- Clang no longer accepts invalid integer constants which are too large to fit
  into any (standard or extended) integer type when the constant is unevaluated.
  Merely forming the token is sufficient to render the program invalid. Code
  like this was previously accepted and is now rejected (#GH134658):
  .. code-block:: c

    #if 1 ? 1 : 999999999999999999999
    #endif
- ``#embed`` directive now diagnoses use of a non-character file (device file)
  such as ``/dev/urandom`` as an error. This restriction may be relaxed in the
  future. See (#GH126629).
- Fixed a clang 20 regression where diagnostics attached to some calls to member functions
  using C++23 "deducing this" did not have a diagnostic location (#GH135522)

- Fixed a crash when a ``friend`` function is redefined as deleted. (#GH135506)
- Fixed a crash when ``#embed`` appears as a part of a failed constant
  evaluation. The crashes were happening during diagnostics emission due to
  unimplemented statement printer. (#GH132641)
- Fixed visibility calculation for template functions. (#GH103477)
- Fixed a bug where an attribute before a ``pragma clang attribute`` or
  ``pragma clang __debug`` would cause an assertion. Instead, this now diagnoses
  the invalid attribute location appropriately. (#GH137861)
- Fixed a crash when a malformed ``_Pragma`` directive appears as part of an
  ``#include`` directive. (#GH138094)
- Fixed a crash during constant evaluation involving invalid lambda captures
  (#GH138832)
- Fixed a crash when instantiating an invalid dependent friend template specialization.
  (#GH139052)
- Fixed a crash with an invalid member function parameter list with a default
  argument which contains a pragma. (#GH113722)
- Fixed assertion failures when generating name lookup table in modules. (#GH61065, #GH134739)
- Fixed an assertion failure in constant compound literal statements. (#GH139160)
- Fix crash due to unknown references and pointer implementation and handling of
  base classes. (GH139452)
- Fixed an assertion failure in serialization of constexpr structs containing unions. (#GH140130)
- Fixed duplicate entries in TableGen that caused the wrong attribute to be selected. (GH#140701)

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The behaviour of ``__add_pointer`` and ``__remove_pointer`` for Objective-C++'s ``id`` and interfaces has been fixed.

- The signature for ``__builtin___clear_cache`` was changed from
  ``void(char *, char *)`` to ``void(void *, void *)`` to match GCC's signature
  for the same builtin. (#GH47833)

- ``__has_unique_object_representations(Incomplete[])`` is no longer accepted, per
  `LWG4113 <https://cplusplus.github.io/LWG/issue4113>`_.

- ``__builtin_is_cpp_trivially_relocatable``, ``__builtin_is_replaceable`` and
  ``__builtin_trivially_relocate`` have been added to support standard C++26 relocation.

- ``__is_trivially_relocatable`` has been deprecated, and uses should be replaced by
  ``__builtin_is_cpp_trivially_relocatable``.
  Note that, it is generally unsafe to ``memcpy`` non-trivially copyable types that
  are ``__builtin_is_cpp_trivially_relocatable``. It is recommended to use
  ``__builtin_trivially_relocate`` instead.

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 - Fixed crash when a parameter to the ``clang::annotate`` attribute evaluates to ``void``. See #GH119125

- Clang now emits a warning instead of an error when using the one or two
  argument form of GCC 11's ``__attribute__((malloc(deallocator)))``
  or ``__attribute__((malloc(deallocator, ptr-index)))``
  (`#51607 <https://github.com/llvm/llvm-project/issues/51607>`_).

- Corrected the diagnostic for the ``callback`` attribute when passing too many
  or too few attribute argument indicies for the specified callback function.
  (#GH47451)

- No longer crashing on ``__attribute__((align_value(N)))`` during template
  instantiation when the function parameter type is not a pointer or reference.
  (#GH26612)
- Now allowing the ``[[deprecated]]``, ``[[maybe_unused]]``, and
  ``[[nodiscard]]`` to be applied to a redeclaration after a definition in both
  C and C++ mode for the standard spellings (other spellings, such as
  ``__attribute__((unused))`` are still ignored after the definition, though
  this behavior may be relaxed in the future). (#GH135481)

- Clang will warn if a complete type specializes a deprecated partial specialization.
  (#GH44496)

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

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

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed type checking when a statement expression ends in an l-value of atomic type. (#GH106576)
- Fixed uninitialized use check in a lambda within CXXOperatorCallExpr. (#GH129198)
- Fixed a malformed printout of ``CXXParenListInitExpr`` in certain contexts.

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

- HTML tags in comments that span multiple lines are now parsed correctly by Clang's comment parser. (#GH120843)

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed crash when ``-print-stats`` is enabled in compiling IR files. (#GH131608)

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

- Bump the default code object version to 6. ROCm 6.3 is required to run any program compiled with COV6.

NVPTX Support
^^^^^^^^^^^^^^

Hexagon Support
^^^^^^^^^^^^^^^

-  The default compilation target has been changed from V60 to V68.

X86 Support
^^^^^^^^^^^

- The 256-bit maximum vector register size control was removed from
  `AVX10 whitepaper <https://cdrdv2.intel.com/v1/dl/getContent/784343>_`.
  * Re-target ``m[no-]avx10.1`` to enable AVX10.1 with 512-bit maximum vector register size.
  * Emit warning for ``mavx10.x-256``, noting AVX10/256 is not supported.
  * Emit warning for ``mavx10.x-512``, noting to use ``m[no-]avx10.x`` instead.
  * Emit warning for ``m[no-]evex512``, noting AVX10/256 is not supported.
  * The features avx10.x-256/512 keep unchanged and will be removed in the next release.

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

- Support has been added for the following processors (command-line identifiers in parentheses):
  - Arm Cortex-A320 (``cortex-a320``)
- For ARM targets, cc1as now considers the FPU's features for the selected CPU or Architecture.
- The ``+nosimd`` attribute is now fully supported for ARM. Previously, this had no effect when being used with
  ARM targets, however this will now disable NEON instructions being generated. The ``simd`` option is
  also now printed when the ``--print-supported-extensions`` option is used.
- When a feature that depends on NEON (``simd``) is used, NEON is now automatically enabled.
- When NEON is disabled (``+nosimd``), all features that depend on NEON will now be disabled.

-  Support for __ptrauth type qualifier has been added.

- For AArch64, added support for generating executable-only code sections by using the
  ``-mexecute-only`` or ``-mpure-code`` compiler flags. (#GH125688)

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

- Clang now defines ``_CRT_USE_BUILTIN_OFFSETOF`` macro in MSVC-compatible mode,
  which makes ``offsetof`` provided by Microsoft's ``<stddef.h>`` to be defined
  correctly. (#GH59689)

- Clang now can process the `i128` and `ui128` integral suffixes when MSVC
  extensions are enabled. This allows for properly processing ``intsafe.h`` in
  the Windows SDK.

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^

- Add support for `-mtune=generic-ooo` (a generic out-of-order model).
- Adds support for `__attribute__((interrupt("SiFive-CLIC-preemptible")))` and
  `__attribute__((interrupt("SiFive-CLIC-stack-swap")))`. The former
  automatically saves some interrupt CSRs before re-enabling interrupts in the
  function prolog, the latter swaps `sp` with the value in a CSR before it is
  used or modified. These two can also be combined, and can be combined with
  `interrupt("machine")`.

- Adds support for `__attribute__((interrupt("qci-nest")))` and
  `__attribute__((interrupt("qci-nonest")))`. These use instructions from
  Qualcomm's `Xqciint` extension to save and restore some GPRs in interrupt
  service routines.

- `Zicsr` / `Zifencei` are allowed to be duplicated in the presence of `g` in `-march`.

- Add support for the `__builtin_riscv_pause()` intrinsic from the `Zihintpause` extension.

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

- Ensure ``isDerivedFrom`` matches the correct base in case more than one alias exists.
- Extend ``templateArgumentCountIs`` to support function and variable template
  specialization.

clang-format
------------

- Adds ``BreakBeforeTemplateCloser`` option.
- Adds ``BinPackLongBracedList`` option to override bin packing options in
  long (20 item or more) braced list initializer lists.
- Add the C language instead of treating it like C++.
- Allow specifying the language (C, C++, or Objective-C) for a ``.h`` file by
  adding a special comment (e.g. ``// clang-format Language: ObjC``) near the
  top of the file.
- Add ``EnumTrailingComma`` option for inserting/removing commas at the end of
  ``enum`` enumerator lists.
- Add ``OneLineFormatOffRegex`` option for turning formatting off for one line.
- Add ``SpaceAfterOperatorKeyword`` option.

clang-refactor
--------------
- Reject `0` as column or line number in 1-based command-line source locations.
  Fixes crash caused by `0` input in `-selection=<file>:<line>:<column>[-<line>:<column>]`. (#GH139457)

libclang
--------
- Fixed a bug in ``clang_File_isEqual`` that sometimes led to different
  in-memory files to be considered as equal.
- Added ``clang_visitCXXMethods``, which allows visiting the methods
  of a class.
- Added ``clang_getFullyQualifiedName``, which provides fully qualified type names as
  instructed by a PrintingPolicy.

- Fixed a buffer overflow in ``CXString`` implementation. The fix may result in
  increased memory allocation.

- Deprecate ``clang_Cursor_GetBinaryOpcode`` and ``clang_Cursor_getBinaryOpcodeStr``
  implementations, which are duplicates of ``clang_getCursorBinaryOperatorKind``
  and ``clang_getBinaryOperatorKindSpelling`` respectively.

Code Completion
---------------
- Reject `0` as column or line number in 1-based command-line source locations.
  Fixes crash caused by `0` input in `-code-completion-at=<file>:<line>:<column>`. (#GH139457)

Static Analyzer
---------------
- Fixed a crash when C++20 parenthesized initializer lists are used. This issue
  was causing a crash in clang-tidy. (#GH136041)

New features
^^^^^^^^^^^^

A new flag - `-static-libclosure` was introduced to support statically linking
the runtime for the Blocks extension on Windows. This flag currently only
changes the code generation, and even then, only on Windows. This does not
impact the linker behaviour like the other `-static-*` flags.

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^

Improvements
^^^^^^^^^^^^

- The checker option ``optin.cplusplus.VirtualCall:PureOnly`` was removed,
  because it had been deprecated since 2019 and it is completely useless (it
  was kept only for compatibility with pre-2019 versions, setting it to true is
  equivalent to completely disabling the checker).

Moved checkers
^^^^^^^^^^^^^^

- After lots of improvements, the checker ``alpha.security.ArrayBoundV2`` is
  renamed to ``security.ArrayBound``. As this checker is stable now, the old
  checker ``alpha.security.ArrayBound`` (which was searching for the same kind
  of bugs with an different, simpler and less accurate algorithm) is removed.

.. _release-notes-sanitizers:

Sanitizers
----------

- ``-fsanitize=vptr`` is no longer a part of ``-fsanitize=undefined``.

Python Binding Changes
----------------------
- Made ``Cursor`` hashable.
- Added ``Cursor.has_attrs``, a binding for ``clang_Cursor_hasAttrs``, to check
  whether a cursor has any attributes.
- Added ``Cursor.specialized_template``, a binding for
  ``clang_getSpecializedCursorTemplate``, to retrieve the primary template that
  the cursor is a specialization of.
- Added ``Type.get_methods``, a binding for ``clang_visitCXXMethods``, which
  allows visiting the methods of a class.
- Added ``Type.get_fully_qualified_name``, which provides fully qualified type names as
  instructed by a PrintingPolicy.
- Add equality comparison operators for ``File`` type

OpenMP Support
--------------
- Added support 'no_openmp_constructs' assumption clause.
- Added support for 'self_maps' in map and requirement clause.
- Added support for 'omp stripe' directive.
- Fixed a crashing bug with ``omp unroll partial`` if the argument to
  ``partial`` was an invalid expression. (#GH139267)
- Fixed a crashing bug with ``omp tile sizes`` if the argument to ``sizes`` was
  an invalid expression. (#GH139073)
- Fixed a crashing bug with ``omp simd collapse`` if the argument to
  ``collapse`` was an invalid expression. (#GH138493)
- Fixed a crashing bug with a malformed ``cancel`` directive. (#GH139360)
- Fixed a crashing bug with ``omp distribute dist_schedule`` if the argument to
  ``dist_schedule`` was not strictly positive. (#GH139266)
- Fixed two crashing bugs with a malformed ``metadirective`` directive. One was
  a crash if the next token after ``metadirective`` was a paren, bracket, or
  brace. The other was if the next token after the meta directive was not an
  open parenthesis. (#GH139665)
- An error is now emitted when OpenMP ``collapse`` and ``ordered`` clauses have
  an argument larger than what can fit within a 64-bit integer.

Improvements
^^^^^^^^^^^^

Additional Information

===================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us on the `Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_.
