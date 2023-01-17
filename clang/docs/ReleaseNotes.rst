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

- The ``-Wimplicit-function-declaration`` and ``-Wimplicit-int`` warnings
  now default to an error in C99, C11, and C17. As of C2x,
  support for implicit function declarations and implicit int has been removed,
  and the warning options will have no effect. Specifying ``-Wimplicit-int`` in
  C89 mode will now issue warnings instead of being a noop.

  **NOTE**: We recommend that projects using configure scripts verify that the
  results do not change before/after setting
  ``-Werror=implicit-function-declarations`` or ``-Wimplicit-int`` to avoid
  incompatibility with Clang 16.

- ``-Wincompatible-function-pointer-types`` now defaults to an error in all C
  language modes. It may be downgraded to a warning with
  ``-Wno-error=incompatible-function-pointer-types`` or disabled entirely with
  ``-Wno-incompatible-function-pointer-types``.

  **NOTE:** We recommend that projects using configure scripts verify that the
  results do not change before/after setting
  ``-Werror=incompatible-function-pointer-types`` to avoid incompatibility with
  Clang 16.

  .. code-block:: c

    void func(const int *i);
    void other(void) {
      void (*fp)(int *) = func; // Previously a warning, now a downgradable error.
    }

- Clang now disallows types whose sizes aren't a multiple of their alignments
  to be used as the element type of arrays.

  .. code-block:: c

  typedef char int8_a16 __attribute__((aligned(16)));
  int8_a16 array[4]; // Now diagnosed as the element size not being a multiple of the array alignment.

- When compiling for Windows in MSVC compatibility mode (for example by using
  clang-cl), the compiler will now propagate dllimport/export declspecs in
  explicit specializations of class template member functions (`Issue 54717
  <https://github.com/llvm/llvm-project/issues/54717>`_):

  .. code-block:: c++

    template <typename> struct __declspec(dllexport) S {
      void f();
    };
    template<> void S<int>::f() {}  // clang-cl will now dllexport this.

  This matches what MSVC does, so it improves compatibility, but it can also
  cause errors for code which clang-cl would previously accept, for example:

  .. code-block:: c++

    template <typename> struct __declspec(dllexport) S {
      void f();
    };
    template<> void S<int>::f() = delete;  // Error: cannot delete dllexport function.

  .. code-block:: c++

    template <typename> struct __declspec(dllimport) S {
      void f();
    };
    template<> void S<int>::f() {};  // Error: cannot define dllimport function.

  These errors also match MSVC's behavior.

- Clang now diagnoses indirection of ``void *`` in C++ mode as a warning which
  defaults to an error. This is compatible with ISO C++, GCC, ICC, and MSVC. This
  is also now a SFINAE error so constraint checking and SFINAE checking can be
  compatible with other compilers. It is expected that this will be upgraded to
  an error-only diagnostic in the next Clang release.

  .. code-block:: c++

    void func(void *p) {
      *p; // Now diagnosed as a warning-as-error.
    }

- Clang now diagnoses use of a bit-field as an instruction operand in Microsoft
  style inline asm blocks as an error. Previously, a bit-field operand yielded
  the address of the allocation unit the bit-field was stored within; reads or
  writes therefore had the potential to read or write nearby bit-fields. This
  change fixes `issue 57791 <https://github.com/llvm/llvm-project/issues/57791>`_.

  .. code-block:: c++

    typedef struct S {
      unsigned bf:1;
    } S;
    void f(S s) {
      __asm {
        mov eax, s.bf // Now diagnosed as an error.
        mov s.bf, eax // Now diagnosed as an error.
      }
    }

- The ``-fexperimental-new-pass-manager`` and ``-fno-legacy-pass-manager``
  flags have been removed. These have been no-ops since 15.0.0.

- As a side effect of implementing DR692/DR1395/DR1432, Clang now rejects some
  overloaded function templates as ambiguous when one of the candidates has a
  trailing parameter pack.

  .. code-block:: c++

    template <typename T> void g(T, T = T());
    template <typename T, typename... U> void g(T, U...);
    void h() {
      // This is rejected due to ambiguity between the pack and the
      // default argument. Only parameters with arguments are considered during
      // partial ordering of function templates.
      g(42);
    }

- Clang's resource dir used to include the full clang version. It will now
  include only the major version. The new resource directory is
  ``$prefix/lib/clang/$CLANG_MAJOR_VERSION`` and can be queried using
  ``clang -print-resource-dir``, just like before.

- To match GCC, ``__ppc64__`` is no longer defined on PowerPC64 targets. Use
  ``__powerpc64__`` instead.

- ``-p`` is rejected for all targets which are not AIX or OpenBSD. ``-p`` led
  to an ``-Wunused-command-line-argument`` warning in previous releases.

- Clang now diagnoses non-inline externally-visible definitions in C++
  standard header units as per ``[module.import/6]``.  Previously, in Clang-15,
  these definitions were allowed.  Note that such definitions are ODR
  violations if the header is included more than once.

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
- ``stdatomic.h`` will use the internal declarations when targeting pre-C++-23
  on Windows platforms as the MSVC support requires newer C++ standard.
- Correct ``_Static_assert`` to accept the same set of extended integer
  constant expressions as is accpted in other contexts that accept them.
  This fixes `Issue 57687 <https://github.com/llvm/llvm-project/issues/57687>`_.
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
- Fix `-Wpre-c++17-compat` crashing Clang when compiling C++20 code which
  contains deduced template specializations. This Fixes
  `Issue 57369 <https://github.com/llvm/llvm-project/issues/57369>`_
  `Issue 57643 <https://github.com/llvm/llvm-project/issues/57643>`_
  `Issue 57793 <https://github.com/llvm/llvm-project/issues/57793>`_
- Respect constructor constraints during class template argument deduction (CTAD).
  This is the suggested resolution to CWG DR2628.
  `Issue 57646 <https://github.com/llvm/llvm-project/issues/57646>`_
  `Issue 43829 <https://github.com/llvm/llvm-project/issues/43829>`_
- Fixed a crash in C++20 mode in Clang and Clangd when compile source
  with compilation errors.
  `Issue 53628 <https://github.com/llvm/llvm-project/issues/53628>`_
- The template arguments of a variable template being accessed as a
  member will now be represented in the AST.
- Fix incorrect handling of inline builtins with asm labels.
- Finished implementing C++ DR2565, which results in a requirement becoming
  not satisfied in the event of an instantiation failures in a requires expression's
  parameter list. We previously handled this correctly in a constraint evaluation
  context, but not in a requires clause evaluated as a boolean.
- Address the thread identification problems in coroutines.
  `Issue 47177 <https://github.com/llvm/llvm-project/issues/47177>`_
  `Issue 47179 <https://github.com/llvm/llvm-project/issues/47179>`_
- Fix a crash upon stray coloncolon token in C2x mode.
- Reject non-type template arguments formed by casting a non-zero integer
  to a pointer in pre-C++17 modes, instead of treating them as null
  pointers.
- Fix template arguments of pointer and reference not taking the type as
  part of their identity.
  `Issue 47136 <https://github.com/llvm/llvm-project/issues/47136>`_
- Fix a crash when trying to form a recovery expression on a call inside a
  constraint, which re-evaluated the same constraint.
  `Issue 53213 <https://github.com/llvm/llvm-project/issues/53213>`_
  `Issue 45736 <https://github.com/llvm/llvm-project/issues/45736>`_
- Fix an issue when performing constraints partial ordering on non-template
  functions. `Issue 56154 <https://github.com/llvm/llvm-project/issues/56154>`_
- Fix handling of unexpanded packs in template argument expressions.
  `Issue 58679 <https://github.com/llvm/llvm-project/issues/58679>`_
- Fix a crash when a ``btf_type_tag`` attribute is applied to the pointee of
  a function pointer.
- Fix a number of recursively-instantiated constraint issues, which would possibly
  result in a stack overflow.
  `Issue 44304 <https://github.com/llvm/llvm-project/issues/44304>`_
  `Issue 50891 <https://github.com/llvm/llvm-project/issues/50891>`_
- Clang 14 predeclared some builtin POSIX library functions in ``gnu2x`` mode,
  and Clang 15 accidentally stopped predeclaring those functions in that
  language mode. Clang 16 now predeclares those functions again. This fixes
  `Issue 56607 <https://github.com/llvm/llvm-project/issues/56607>`_.
- GNU attributes being applied prior to standard attributes would be handled
  improperly, which was corrected to match the behaviour exhibited by GCC.
  `Issue 58229 <https://github.com/llvm/llvm-project/issues/58229>`_
- The builtin type trait ``__is_aggregate`` now returns ``true`` for arrays of incomplete
  types in accordance with the suggested fix for `LWG3823 <https://cplusplus.github.io/LWG/issue3823>`_
- Fix bug with using enum that could lead to enumerators being treated as if
  they were part of an overload set. This fixes
  `Issue 58067 <https://github.com/llvm/llvm-project/issues/58057>`_
  `Issue 59014 <https://github.com/llvm/llvm-project/issues/59014>`_
  `Issue 54746 <https://github.com/llvm/llvm-project/issues/54746>`_
- Fix assert that triggers a crash during some types of list initialization that
  generate a CXXTemporaryObjectExpr instead of a InitListExpr. This fixes
  `Issue 58302 <https://github.com/llvm/llvm-project/issues/58302>`_
  `Issue 58753 <https://github.com/llvm/llvm-project/issues/58753>`_
  `Issue 59100 <https://github.com/llvm/llvm-project/issues/59100>`_
- Fix issue using __attribute__((format)) on non-variadic functions that expect
  more than one formatted argument.
- Fix bug where constant evaluation treated a pointer to member that points to
  a weak member as never being null. Such comparisons are now treated as
  non-constant.
- Fix sanity check when value initializing an empty union so that it takes into
  account anonymous structs which is a GNU extension. This fixes
  `Issue 58800 <https://github.com/llvm/llvm-project/issues/58800>`_
- Fix an issue that triggers a crash if we instantiate a hidden friend functions.
  This fixes `Issue 54457 <https://github.com/llvm/llvm-project/issues/54457>`_
- Fix an issue where -frewrite-includes generated line control directives with
  incorrect line numbers in some cases when a header file used an end of line
  character sequence that differed from the primary source file.
  `Issue 59736 <https://github.com/llvm/llvm-project/issues/59736>`_
- In C mode, when ``e1`` has ``__attribute__((noreturn))`` but ``e2`` doesn't,
  ``(c ? e1 : e2)`` is no longer considered noreturn.
  `Issue 59792 <https://github.com/llvm/llvm-project/issues/59792>`_
- Fix an issue that makes Clang crash on lambda template parameters. This fixes
  `Issue 57960 <https://github.com/llvm/llvm-project/issues/57960>`_
- Fix issue that the standard C++ modules importer will call global
  constructor/destructor for the global varaibles in the importing modules.
  This fixes `Issue 59765 <https://github.com/llvm/llvm-project/issues/59765>`_
- Reject in-class defaulting of previosly declared comparison operators. Fixes
  `Issue 51227 <https://github.com/llvm/llvm-project/issues/51227>`_.

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
- Clang now correctly diagnoses index that refers past the last possible element
  of FAM-like arrays.
- Clang now correctly diagnoses a warning when defercencing a void pointer in C mode.
  This fixes `Issue 53631 <https://github.com/llvm/llvm-project/issues/53631>`_
- Clang will now diagnose an overload set where a candidate has a constraint that
  refers to an expression with a previous error as nothing viable, so that it
  doesn't generate strange cascading errors, particularly in cases where a
  subsuming constraint fails, which would result in a less-specific overload to
  be selected.
- Add a fix-it hint for the ``-Wdefaulted-function-deleted`` warning to
  explicitly delete the function.
- Fixed an accidental duplicate diagnostic involving the declaration of a
  function definition without a prototype which is preceded by a static
  declaration of the function with a prototype. Fixes
  `Issue 58181 <https://github.com/llvm/llvm-project/issues/58181>`_.
- Copy-elided initialization of lock scopes is now handled differently in
  ``-Wthread-safety-analysis``: annotations on the move constructor are no
  longer taken into account, in favor of annotations on the function returning
  the lock scope by value. This could result in new warnings if code depended
  on the previous undocumented behavior. As a side effect of this change,
  constructor calls outside of initializer expressions are no longer ignored,
  which can result in new warnings (or make existing warnings disappear).
- The wording of diagnostics regarding arithmetic on fixed-sized arrays and
  pointers is improved to include the type of the array and whether it's cast
  to another type. This should improve comprehension for why an index is
  out-of-bounds.
- Clang now correctly points to the problematic parameter for the ``-Wnonnull``
  warning. This fixes
  `Issue 58273 <https://github.com/llvm/llvm-project/issues/58273>`_.
- Introduced ``-Wcast-function-type-strict`` and
  ``-Wincompatible-function-pointer-types-strict`` to warn about function type
  mismatches in casts and assignments that may result in runtime indirect call
  `Control-Flow Integrity (CFI)
  <https://clang.llvm.org/docs/ControlFlowIntegrity.html>`_ failures. The
  ``-Wcast-function-type-strict`` diagnostic is grouped under
  ``-Wcast-function-type`` as it identifies a more strict set of potentially
  problematic function type casts.
- Clang will now disambiguate NTTP types when printing diagnostic that contain NTTP types.
  Fixes `Issue 57562 <https://github.com/llvm/llvm-project/issues/57562>`_.
- Better error recovery for pack expansion of expressions.
  `Issue 58673 <https://github.com/llvm/llvm-project/issues/58673>`_.
- Better diagnostics when the user has missed `auto` in a declaration.
  `Issue 49129 <https://github.com/llvm/llvm-project/issues/49129>`_.
- Clang now diagnoses use of invalid or reserved module names in a module
  export declaration. Both are diagnosed as an error, but the diagnostic is
  suppressed for use of reserved names in a system header.
- ``-Winteger-overflow`` will diagnose overflow in more cases. This fixes
  `Issue 58944 <https://github.com/llvm/llvm-project/issues/58944>`_.
- Clang has an internal limit of 2GB of preprocessed source code per
  compilation, including source reachable through imported AST files such as
  PCH or modules. When Clang hits this limit, it now produces notes mentioning
  which header and source files are consuming large amounts of this space.
  ``#pragma clang __debug sloc_usage`` can also be used to request this report.
- Clang no longer permits the keyword 'bool' in a concept declaration as a
  concepts-ts compatibility extension.
- Clang now diagnoses overflow undefined behavior in a constant expression while
  evaluating a compound assignment with remainder as operand.
- Add ``-Wreturn-local-addr``, a GCC alias for ``-Wreturn-stack-address``.
- Clang now suppresses ``-Wlogical-op-parentheses`` on ``(x && a || b)`` and ``(a || b && x)``
  only when ``x`` is a string literal.
- Clang will now reject the GNU extension address of label in coroutines explicitly.
  This fixes `Issue 56436 <https://github.com/llvm/llvm-project/issues/56436>`_.

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
- Unicode support has been updated to support Unicode 15.0.
  New unicode codepoints are supported as appropriate in diagnostics,
  C and C++ identifiers, and escape sequences.
- In identifiers, Clang allows a restricted set of additional mathematical symbols
  as an extension. These symbols correspond to a proposed Unicode
  `Mathematical notation profile for default identifiers
  <https://www.unicode.org/L2/L2022/22230-math-profile.pdf>`_.
  This resolves `Issue 54732 <https://github.com/llvm/llvm-project/issues/54732>`_.
- Clang now supports loading multiple configuration files. The files from
  default configuration paths are loaded first, unless ``--no-default-config``
  option is used. All files explicitly specified using ``--config=`` option
  are loaded afterwards.
- When loading default configuration files, clang now unconditionally uses
  the real target triple (respecting options such as ``--target=`` and ``-m32``)
  rather than the executable prefix. The respective configuration files are
  also loaded when clang is called via an executable without a prefix (e.g.
  plain ``clang``).
- Default configuration paths were partially changed. Clang now attempts to load
  ``<triple>-<driver>.cfg`` first, and falls back to loading both
  ``<driver>.cfg`` and ``<triple>.cfg`` if the former is not found. `Triple`
  is the target triple and `driver` first tries the canonical name
  for the driver (respecting ``--driver-mode=``), and then the name found
  in the executable.
- If the environment variable ``SOURCE_DATE_EPOCH`` is set, it specifies a UNIX
  timestamp to be used in replacement of the current date and time in
  the ``__DATE__``, ``__TIME__``, and ``__TIMESTAMP__`` macros. See
  `<https://reproducible-builds.org/docs/source-date-epoch/>`_.
- Clang now supports ``__has_constexpr_builtin`` function-like macro that
  evaluates to 1 if the builtin is supported and can be constant evaluated.
  It can be used to writing conditionally constexpr code that uses builtins.
- The time profiler (using ``-ftime-trace`` option) now traces various constant
  evaluation events.
- Clang can now generate a PCH when using ``-fdelayed-template-parsing`` for
  code with templates containing loop hint pragmas, OpenMP pragmas, and
  ``#pragma unused``.
- Now diagnoses use of a member access expression or array subscript expression
  within ``__builtin_offsetof`` and ``offsetof`` as being a Clang extension.

New Compiler Flags
------------------

- Implemented `-fcoro-aligned-allocation` flag. This flag implements
  Option 2 of P2014R0 aligned allocation of coroutine frames
  (`P2014R0 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2014r0.pdf>`_).
  With this flag, the coroutines will try to lookup aligned allocation
  function all the time. The compiler will emit an error if it fails to
  find aligned allocation function. So if the user code implemented self
  defined allocation function for coroutines, the existing code will be
  broken. A little divergence with P2014R0 is that clang will lookup
  `::operator new(size_­t, std::aligned_val_t, nothrow_­t)` if there is
  `get_­return_­object_­on_­allocation_­failure`. We feel this is more consistent
  with the intention.

- Added ``--no-default-config`` to disable automatically loading configuration
  files using default paths.

- Added the new level, ``3``, to the ``-fstrict-flex-arrays=`` flag. The new
  level is the strict, standards-conforming mode for flexible array members. It
  recognizes only incomplete arrays as flexible array members (which is how the
  feature is defined by the C standard).

  .. code-block:: c

    struct foo {
      int a;
      int b[]; // Flexible array member.
    };

    struct bar {
      int a;
      int b[0]; // NOT a flexible array member.
    };

- Added ``-fmodule-output`` to enable the one-phase compilation model for
  standard C++ modules. See
  `Standard C++ Modules <https://clang.llvm.org/docs/StandardCPlusPlusModules.html>`_
  for more information.

Deprecated Compiler Flags
-------------------------
- ``-enable-trivial-auto-var-init-zero-knowing-it-will-be-removed-from-clang``
  has been deprecated. The flag will be removed in Clang 18.
  ``-ftrivial-auto-var-init=zero`` is now available unconditionally, to be
  compatible with GCC.

Modified Compiler Flags
-----------------------
- Clang now permits specifying ``--config=`` multiple times, to load multiple
  configuration files.

Removed Compiler Flags
-------------------------
- Clang now no longer supports ``-cc1 -fconcepts-ts``.  This flag has been deprecated
  and encouraged use of ``-std=c++20`` since Clang 10, so we're now removing it.

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

- Introduced a new record declaration attribute ``__attribute__((enforce_read_only_placement))``
  to support analysis of instances of a given type focused on read-only program
  memory placement. It emits a warning if something in the code provably prevents
  an instance from a read-only memory placement.

Windows Support
---------------
- For the MinGW driver, added the options ``-mguard=none``, ``-mguard=cf`` and
  ``-mguard=cf-nochecks`` (equivalent to ``/guard:cf-``, ``/guard:cf`` and
  ``/guard:cf,nochecks`` in clang-cl) for enabling Control Flow Guard checks
  and generation of address-taken function table.

- Switched from SHA1 to BLAKE3 for PDB type hashing / ``-gcodeview-ghash``

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
- Implemented `WG14 N2508 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2508.pdf>`_,
  so labels can placed everywhere inside a compound statement.
- Implemented `WG14 N2927 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2927.htm>`_,
  the Not-so-magic ``typeof`` operator. Also implemented
  `WG14 N2930 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2930.pdf>`_,
  renaming ``remove_quals``, so the ``typeof_unqual`` operator is also
  supported. Both of these operators are supported only in C2x mode. The
  ``typeof`` operator specifies the type of the given parenthesized expression
  operand or type name, including all qualifiers. The ``typeof_unqual``
  operator is similar to ``typeof`` except that all qualifiers are removed,
  including atomic type qualification and type attributes which behave like a
  qualifier, such as an address space attribute.

  .. code-block:: c

    __attribute__((address_space(1))) const _Atomic int Val;
    typeof(Val) OtherVal; // type is '__attribute__((address_space(1))) const _Atomic int'
    typeof_unqual(Val) OtherValUnqual; // type is 'int'

- Implemented `WG14 N3042 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3042.htm>`_,
  Introduce the nullptr constant. This introduces a new type ``nullptr_t``,
  declared in ``<stddef.h>`` which represents the type of the null pointer named
  constant, ``nullptr``. This constant is implicitly convertible to any pointer
  type and represents a type-safe null value.

  Note, there are some known incompatibilities with this same feature in C++.
  The following examples were discovered during implementation and are subject
  to change depending on how national body comments are resolved by WG14 (C
  status is based on standard requirements, not necessarily implementation
  behavior):

  .. code-block:: c

    nullptr_t null_val;
    (nullptr_t)nullptr;       // Rejected in C, accepted in C++, Clang accepts
    (void)(1 ? nullptr : 0);  // Rejected in C, accepted in C++, Clang rejects
    (void)(1 ? null_val : 0); // Rejected in C, accepted in C++, Clang rejects
    bool b1 = nullptr;        // Accepted in C, rejected in C++, Clang rejects
    b1 = null_val;            // Accepted in C, rejected in C++, Clang rejects
    null_val = 0;             // Rejected in C, accepted in C++, Clang rejects

    void func(nullptr_t);
    func(0);                  // Rejected in C, accepted in C++, Clang rejects

- Implemented `WG14 N2975 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2975.pdf>`_,
  Relax requirements for va_start. In C2x mode, functions can now be declared
  fully variadic and the ``va_start`` macro no longer requires passing a second
  argument (though it accepts a second argument for backwards compatibility).
  If a second argument is passed, it is neither expanded nor evaluated in C2x
  mode.

  .. code-block:: c

    void func(...) {  // Invalid in C17 and earlier, valid in C2x and later.
      va_list list;
      va_start(list); // Invalid in C17 and earlier, valid in C2x and later.
      va_end(list);
    }

- Diagnose type definitions in the ``type`` argument of ``__builtin_offsetof``
  as a conforming C extension according to
  `WG14 N2350 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2350.htm>`_.
  Also documents the builtin appropriately. Note, a type definition in C++
  continues to be rejected.

C++ Language Changes in Clang
-----------------------------
- Implemented `DR692 <https://wg21.link/cwg692>`_, `DR1395 <https://wg21.link/cwg1395>`_,
  and `DR1432 <https://wg21.link/cwg1432>`_. The fix for DR1432 is speculative since the
  issue is still open and has no proposed resolution at this time. A speculative fix
  for DR1432 is needed to prevent regressions that would otherwise occur due to DR692.
- Clang's default C++/ObjC++ standard is now ``gnu++17`` instead of ``gnu++14``.
  This means Clang will by default accept code using features from C++17 and
  conforming GNU extensions. Projects incompatible with C++17 can add
  ``-std=gnu++14`` to their build settings to restore the previous behaviour.
- Implemented DR2358 allowing init captures in lambdas in default arguments.
- implemented `DR2654 <https://wg21.link/cwg2654>`_ which undeprecates
  all compound assignements operations on volatile qualified variables.
- Implemented DR2631. Invalid ``consteval`` calls in default arguments and default
  member initializers are diagnosed when and if the default is used.
  This Fixes `Issue 56379 <https://github.com/llvm/llvm-project/issues/56379>`_
  and changes the value of ``std::source_location::current()``
  used in default parameters calls compared to previous versions of Clang.

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Support capturing structured bindings in lambdas
  (`P1091R3 <https://wg21.link/p1091r3>`_ and `P1381R1 <https://wg21.link/P1381R1>`_).
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
- Clang now correctly delays the instantiation of function constraints until
  the time of checking, which should now allow the libstdc++ ranges implementation
  to work for at least trivial examples.  This fixes
  `Issue 44178 <https://github.com/llvm/llvm-project/issues/44178>`_.
- Clang implements DR2621, correcting a defect in ``using enum`` handling.  The
  name is found via ordinary lookup so typedefs are found.
- Implemented `P0634r3 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0634r3.html>`_,
  which removes the requirement for the ``typename`` keyword in certain contexts.
- Implemented The Equality Operator You Are Looking For (`P2468 <http://wg21.link/p2468r2>`_).
- Implemented `P2113R0: Proposed resolution for 2019 comment CA 112 <https://wg21.link/P2113R0>`_
  ([temp.func.order]p6.2.1 is not implemented, matching GCC).
- Implemented `P0857R0 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0857r0.html>`_,
  which specifies constrained lambdas and constrained template *template-parameter*\s.
- Required parameter pack to be provided at the end of the concept parameter list. This
  fixes `Issue 48182 <https://github.com/llvm/llvm-project/issues/48182>`_.

- Do not hide templated base members introduced via using-decl in derived class
  (useful specially for constrained members). Fixes `GH50886 <https://github.com/llvm/llvm-project/issues/50886>`_.
- Implemented CWG2635 as a Defect Report, which prohibits structured bindings from being constrained.
- Correctly handle access-checks in requires expression. Fixes `GH53364 <https://github.com/llvm/llvm-project/issues/53364>`_,
  `GH53334 <https://github.com/llvm/llvm-project/issues/53334>`_.

- Implemented `P0960R3: <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0960r3.html>`_
  and `P1975R0: <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1975r0.html>`_,
  which allows parenthesized aggregate-initialization.

C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Support label at end of compound statement (`P2324 <https://wg21.link/p2324r2>`_).
- Implemented `P1169R4: static operator() <https://wg21.link/P1169R4>`_ and `P2589R1: static operator[] <https://wg21.link/P2589R1>`_.
- Implemented "char8_t Compatibility and Portability Fix" (`P2513R3 <https://wg21.link/P2513R3>`_).
  This change was applied to C++20 as a Defect Report.
- Implemented "Permitting static constexpr variables in constexpr functions" (`P2647R1 <https://wg21.link/P2647R1>_`).
- Implemented `CWG2640 Allow more characters in an n-char sequence <https://wg21.link/CWG2640>_`.

CUDA/HIP Language Changes in Clang
----------------------------------

 - Allow the use of ``__noinline__`` as a keyword (instead of ``__attribute__((noinline))``)
   in lambda declarations.

Objective-C Language Changes in Clang
-------------------------------------

OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------

- GCC doesn't pack non-POD members in packed structs unless the packed
  attribute is also specified on the member. Clang historically did perform
  such packing. Clang now matches the gcc behavior (except on Darwin and PS4).
  You can switch back to the old ABI behavior with the flag:
  ``-fclang-abi-compat=15.0``.
- GCC allows POD types to have defaulted special members. Clang historically
  classified such types as non-POD (for the purposes of Itanium ABI). Clang now
  matches the gcc behavior (except on Darwin and PS4). You can switch back to
  the old ABI behavior with the flag: ``-fclang-abi-compat=15.0``.

OpenMP Support in Clang
-----------------------

...

CUDA Support in Clang
---------------------

- Clang now supports CUDA SDK up to 11.8
- Added support for targeting sm_{87,89,90} GPUs.

LoongArch Support in Clang
--------------------------
- Clang now supports LoongArch. Along with the backend, clang is able to build a
  large corpus of Linux applications. Test-suite 100% pass.
- Support basic option ``-march=`` which is used to select the target
  architecture, i.e. the basic set of ISA modules to be enabled. Possible values
  are ``loongarch64`` and ``la464``.
- Support basic option ``-mabi=`` which is used to select the base ABI type.
  Possible values are ``lp64d``, ``lp64f``, ``lp64s``, ``ilp32d``, ``ilp32f``
  and ``ilp32s``.
- Support extended options: ``-msoft-float``, ``-msingle-float``, ``-mdouble-float`` and ``mfpu=``.
  See `LoongArch toolchain conventions <https://loongson.github.io/LoongArch-Documentation/LoongArch-toolchain-conventions-EN.html>`_.

RISC-V Support in Clang
-----------------------
- ``sifive-7-rv32`` and ``sifive-7-rv64`` are no longer supported for ``-mcpu``.
  Use ``sifive-e76``, ``sifive-s76``, or ``sifive-u74`` instead.
- Native detections via ``-mcpu=native`` and ``-mtune=native`` are supported.
- Fix interaction of ``-mcpu`` and ``-march``, RISC-V backend will take the
  architecture extension union of ``-mcpu`` and ``-march`` before, and now will
  take architecture extensions from ``-march`` if both are given.

X86 Support in Clang
--------------------
- Support ``-mindirect-branch-cs-prefix`` for call and jmp to indirect thunk.
- Fix 32-bit ``__fastcall`` and ``__vectorcall`` ABI mismatch with MSVC.
- Add ISA of ``AMX-FP16`` which support ``_tile_dpfp16ps``.
- Switch ``AVX512-BF16`` intrinsics types from ``short`` to ``__bf16``.
- Add support for ``PREFETCHI`` instructions.
- Support ISA of ``CMPCCXADD``.
  * Support intrinsic of ``_cmpccxadd_epi32``.
  * Support intrinsic of ``_cmpccxadd_epi64``.
- Add support for ``RAO-INT`` instructions.
  * Support intrinsic of ``_aadd_i32/64``
  * Support intrinsic of ``_aand_i32/64``
  * Support intrinsic of ``_aor_i32/64``
  * Support intrinsic of ``_axor_i32/64``
- Support ISA of ``AVX-IFMA``.
  * Support intrinsic of ``_mm(256)_madd52hi_avx_epu64``.
  * Support intrinsic of ``_mm(256)_madd52lo_avx_epu64``.
- Support ISA of ``AVX-VNNI-INT8``.
  * Support intrinsic of ``_mm(256)_dpbssd(s)_epi32``.
  * Support intrinsic of ``_mm(256)_dpbsud(s)_epi32``.
  * Support intrinsic of ``_mm(256)_dpbuud(s)_epi32``.
- Support ISA of ``AVX-NE-CONVERT``.
  * Support intrinsic of ``_mm(256)_bcstnebf16_ps``.
  * Support intrinsic of ``_mm(256)_bcstnesh_ps``.
  * Support intrinsic of ``_mm(256)_cvtneebf16_ps``.
  * Support intrinsic of ``_mm(256)_cvtneeph_ps``.
  * Support intrinsic of ``_mm(256)_cvtneobf16_ps``.
  * Support intrinsic of ``_mm(256)_cvtneoph_ps``.
  * Support intrinsic of ``_mm(256)_cvtneps_avx_pbh``.
- ``-march=raptorlake``, ``-march=meteorlake`` and ``-march=emeraldrapids`` are now supported.
- ``-march=sierraforest``, ``-march=graniterapids`` and ``-march=grandridge`` are now supported.
- Lift _BitInt() supported max width from 128 to 8388608.
- Support intrinsics of ``_mm(256)_reduce_(add|mul|or|and)_epi8/16``.
- Support intrinsics of ``_mm(256)_reduce_(max|min)_ep[i|u]8/16``.

WebAssembly Support in Clang
----------------------------

The -mcpu=generic configuration now enables sign-ext and mutable-globals. These
proposals are standardized and available in all major engines.

DWARF Support in Clang
----------------------

Previously when emitting DWARFv4 and tuning for GDB, Clang would use DWARF v2's
``DW_AT_bit_offset`` and ``DW_AT_data_member_location``. Clang now uses DWARF v4's
``DW_AT_data_bit_offset`` regardless of tuning.

Support for ``DW_AT_data_bit_offset`` was added in GDB 8.0. For earlier versions,
you can use the ``-gdwarf-3`` option to emit compatible DWARF.

Arm and AArch64 Support in Clang
--------------------------------

- The target(..) function attributes for AArch64 now accept:

  * ``"arch=<arch>"`` strings, that specify the architecture for a function as per the ``-march`` option.
  * ``"cpu=<cpu>"`` strings, that specify the cpu for a function as per the ``-mcpu`` option.
  * ``"tune=<cpu>"`` strings, that specify the tune cpu for a function as per ``-mtune``.
  * ``"+<feature>"``, ``"+no<feature>"`` enables/disables the specific feature, for compatibility with GCC target attributes.
  * ``"<feature>"``, ``"no-<feature>"`` enabled/disables the specific feature, for backward compatibility with previous releases.
- ``-march`` values for targeting armv2, armv2A, armv3 and armv3M have been removed.
  Their presence gave the impression that Clang can correctly generate code for
  them, which it cannot.
- Support has been added for the following processors (-mcpu identifiers in parenthesis):

  * Arm Cortex-A715 (cortex-a715).
  * Arm Cortex-X3 (cortex-x3).
  * Arm Neoverse V2 (neoverse-v2)
- Strict floating point has been enabled for AArch64, which means that
  ``-ftrapping-math``, ``-frounding-math``, ``-ffp-model=strict``, and
  ``-ffp-exception-behaviour=<arg>`` are now accepted.

Floating Point Support in Clang
-------------------------------
- The driver option ``-menable-unsafe-fp-math`` has been removed. To enable
  unsafe floating-point optimizations use ``-funsafe-math-optimizations`` or
  ``-ffast-math`` instead.
- Add ``__builtin_elementwise_sin`` and ``__builtin_elementwise_cos`` builtins for floating point types only.

Internal API Changes
--------------------

Build System Changes
--------------------

AST Matchers
------------
- Add ``isInAnoymousNamespace`` matcher to match declarations in an anonymous namespace.

clang-format
------------
- Add ``RemoveSemicolon`` option for removing ``;`` after a non-empty function definition.
- Add ``RequiresExpressionIndentation`` option for configuring the alignment of requires-expressions.
  The default value of this option is ``OuterScope``, which differs in behavior from clang-format 15.
  To match the default behavior of clang-format 15, use the ``Keyword`` value.
- Add ``IntegerLiteralSeparator`` option for fixing integer literal separators
  in C++, C#, Java, and JavaScript.
- Add ``BreakAfterAttributes`` option for breaking after a group of C++11
  attributes before a function declaration/definition name.
- Add ``InsertNewlineAtEOF`` option for inserting a newline at EOF if missing.
- Add ``LineEnding`` option to deprecate ``DeriveLineEnding`` and ``UseCRLF``.

clang-extdef-mapping
--------------------

libclang
--------
- Introduced the new function ``clang_getUnqualifiedType``, which mimics
  the behavior of ``QualType::getUnqualifiedType`` for ``CXType``.
- Introduced the new function ``clang_getNonReferenceType``, which mimics
  the behavior of ``QualType::getNonReferenceType`` for ``CXType``.
- Introduced the new function ``clang_CXXMethod_isDeleted``, which queries
  whether the method is declared ``= delete``.
- Introduced the new function ``clang_CXXMethod_isCopyAssignmentOperator``,
  which identifies whether a method cursor is a copy-assignment
  operator.
- Introduced the new function ``clang_CXXMethod_isMoveAssignmentOperator``,
  which identifies whether a method cursor is a move-assignment
  operator.
- ``clang_Cursor_getNumTemplateArguments``, ``clang_Cursor_getTemplateArgumentKind``,
  ``clang_Cursor_getTemplateArgumentType``, ``clang_Cursor_getTemplateArgumentValue`` and
  ``clang_Cursor_getTemplateArgumentUnsignedValue`` now work on struct, class,
  and partial template specialization cursors in addition to function cursors.

Static Analyzer
---------------
- Removed the deprecated ``-analyzer-store`` and
  ``-analyzer-opt-analyze-nested-blocks`` analyzer flags.
  ``scanbuild`` was also updated accordingly.
  Passing these flags will result in a hard error.

- Deprecate the ``consider-single-element-arrays-as-flexible-array-members``
  analyzer-config option.
  This option will be still accepted, but a warning will be displayed.
  This option will be rejected, thus turned into a hard error starting with
  ``clang-17``. Use ``-fstrict-flex-array=<N>`` instead if necessary.

- Trailing array objects of structs with single elements will be considered
  as flexible-array-members. Use ``-fstrict-flex-array=<N>`` to define
  what should be considered as flexible-array-member if needed.

.. _release-notes-sanitizers:

Sanitizers
----------
- ``-fsanitize-memory-param-retval`` is turned on by default. With
  ``-fsanitize=memory``, passing uninitialized variables to functions and
  returning uninitialized variables from functions is more aggressively
  reported. ``-fno-sanitize-memory-param-retval`` restores the previous
  behavior.

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
