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

C/C++ Language Potentially Breaking Changes
-------------------------------------------

C++ Specific Potentially Breaking Changes
-----------------------------------------

- The type trait builtin ``__is_referenceable`` has been removed, since it has
  very few users and all the type traits that could benefit from it in the
  standard library already have their own bespoke builtins.

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

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P1061R10 Structured Bindings can introduce a Pack <https://wg21.link/P1061R10>`_.

- Implemented `P0963R3 Structured binding declaration as a condition <https://wg21.link/P0963R3>`_.

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

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

C Language Changes
------------------

- Clang now allows an ``inline`` specifier on a typedef declaration of a
  function type in Microsoft compatibility mode. #GH124869
- Clang now allows ``restrict`` qualifier for array types with pointer elements (#GH92847).

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

C23 Feature Support
^^^^^^^^^^^^^^^^^^^
- Added ``__builtin_c23_va_start()`` for compatibility with GCC and to enable
  better diagnostic behavior for the ``va_start()`` macro in C23 and later.
  This also updates the definition of ``va_start()`` in ``<stdarg.h>`` to use
  the new builtin. Fixes #GH124031.
- Implemented `WG14 N2819 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2819.pdf>`_
  which clarified that a compound literal used within a function prototype is
  treated as if the compound literal were within the body rather than at file
  scope.

Non-comprehensive list of changes in this release
-------------------------------------------------

- Support parsing the `cc` operand modifier and alias it to the `c` modifier (#GH127719).
- Added `__builtin_elementwise_exp10`.

New Compiler Flags
------------------

- New option ``-Wundef-true`` added and enabled by default to warn when `true` is used in the C preprocessor without being defined before C23.

- New option ``-fprofile-continuous`` added to enable continuous profile syncing to file (#GH124353, `docs <https://clang.llvm.org/docs/UsersManual.html#cmdoption-fprofile-continuous>`_).
  The feature has `existed <https://clang.llvm.org/docs/SourceBasedCodeCoverage.html#running-the-instrumented-program>`_)
  for a while and this is just a user facing option.

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

- The ARM AArch32 ``-mtp`` option accepts and defaults to ``auto``, a value of ``auto`` uses the best available method of providing the frame pointer supported by the hardware. This matches
  the behavior of ``-mtp`` in gcc. This changes the default behavior for ARM targets that provide the ``TPIDRURO`` register as this will be used instead of a call to the ``__aeabi_read_tp``.
  Programs that use ``__aeabi_read_tp`` but do not use the ``TPIDRURO`` register must use ``-mtp=soft``. Fixes #123864

- The compiler flag `-fbracket-depth` default value is increased from 256 to 2048. (#GH94728)

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
- The ``-Wunnecessary-virtual-specifier`` warning has been added to warn about
  methods which are marked as virtual inside a ``final`` class, and hence can
  never be overridden.

- Improve the diagnostics for chained comparisons to report actual expressions and operators (#GH129069).

- Improve the diagnostics for shadows template parameter to report correct location (#GH129060).

- Improve the ``-Wundefined-func-template`` warning when a function template is not instantiated due to being unreachable in modules.

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

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The behvaiour of ``__add_pointer`` and ``__remove_pointer`` for Objective-C++'s ``id`` and interfaces has been fixed.

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 - Fixed crash when a parameter to the ``clang::annotate`` attribute evaluates to ``void``. See #GH119125

- Clang now emits a warning instead of an error when using the one or two
  argument form of GCC 11's ``__attribute__((malloc(deallocator)))``
  or ``__attribute__((malloc(deallocator, ptr-index)))``
  (`#51607 <https://github.com/llvm/llvm-project/issues/51607>`_).

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Clang now diagnoses copy constructors taking the class by value in template instantiations. (#GH130866)
- Clang is now better at keeping track of friend function template instance contexts. (#GH55509)
- Clang now prints the correct instantiation context for diagnostics suppressed
  by template argument deduction.
- Clang is now better at instantiating the function definition after its use inside
  of a constexpr lambda. (#GH125747)
- The initialization kind of elements of structured bindings
  direct-list-initialized from an array is corrected to direct-initialization.
- Clang no longer crashes when a coroutine is declared ``[[noreturn]]``. (#GH127327)
- Clang now uses the parameter location for abbreviated function templates in ``extern "C"``. (#GH46386)
- Clang will emit an error instead of crash when use co_await or co_yield in
  C++26 braced-init-list template parameter initialization. (#GH78426)
- Fixes matching of nested template template parameters. (#GH130362)
- Correctly diagnoses template template paramters which have a pack parameter
  not in the last position.
- Clang now correctly parses ``if constexpr`` expressions in immediate function context. (#GH123524)
- Fixed an assertion failure affecting code that uses C++23 "deducing this". (#GH130272)
- Clang now properly instantiates destructors for initialized members within non-delegating constructors. (#GH93251)
- Correctly diagnoses if unresolved using declarations shadows template paramters (#GH129411)
- Clang was previously coalescing volatile writes to members of volatile base class subobjects.
  The issue has been addressed by propagating qualifiers during derived-to-base conversions in the AST. (#GH127824)
- Fixed a Clang regression in C++20 mode where unresolved dependent call expressions were created inside non-dependent contexts (#GH122892)
- Clang now emits the ``-Wunused-variable`` warning when some structured bindings are unused
  and the ``[[maybe_unused]]`` attribute is not applied. (#GH125810)

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

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

NVPTX Support
^^^^^^^^^^^^^^

Hexagon Support
^^^^^^^^^^^^^^^

-  The default compilation target has been changed from V60 to V68.

X86 Support
^^^^^^^^^^^

- Disable ``-m[no-]avx10.1`` and switch ``-m[no-]avx10.2`` to alias of 512 bit
  options.
- Change ``-mno-avx10.1-512`` to alias of ``-mno-avx10.1-256`` to disable both
  256 and 512 bit instructions.

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

- Clang now defines ``_CRT_USE_BUILTIN_OFFSETOF`` macro in MSVC-compatible mode,
  which makes ``offsetof`` provided by Microsoft's ``<stddef.h>`` to be defined
  correctly. (#GH59689)

- Clang now can process the `i128` and `ui128` integeral suffixes when MSVC
  extensions are enabled. This allows for properly processing ``intsafe.h`` in
  the Windows SDK.

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^

- Add support for `-mtune=generic-ooo` (a generic out-of-order model).

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
- Move ``ast_matchers::MatchFinder::MatchFinderOptions`` to
  ``ast_matchers::MatchFinderOptions``.
- Add a boolean member ``SkipSystemHeaders`` to ``MatchFinderOptions``, and make
  ``MatchASTConsumer`` receive a reference to ``MatchFinderOptions`` in the
  constructor. This allows it to skip system headers when traversing the AST.

clang-format
------------

- Adds ``BreakBeforeTemplateCloser`` option.
- Adds ``BinPackLongBracedList`` option to override bin packing options in
  long (20 item or more) braced list initializer lists.
- Add the C language instead of treating it like C++.
- Allow specifying the language (C, C++, or Objective-C) for a ``.h`` file by
  adding a special comment (e.g. ``// clang-format Language: ObjC``) near the
  top of the file.

libclang
--------
- Added ``clang_visitCXXMethods``, which allows visiting the methods
  of a class.

- Fixed a buffer overflow in ``CXString`` implementation. The fix may result in
  increased memory allocation.

Code Completion
---------------

Static Analyzer
---------------

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
- Added ``Type.get_methods``, a binding for ``clang_visitCXXMethods``, which
  allows visiting the methods of a class.

OpenMP Support
--------------
- Added support 'no_openmp_constructs' assumption clause.
- Added support for 'self_maps' in map and requirement clause.
- Added support for 'omp stripe' directive.
- Added support for private variable reduction.

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
