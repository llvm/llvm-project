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

- Fix a bug in reversed argument for templated operators.
  This breaks code in C++20 which was previously accepted in C++17.
  Clang did not properly diagnose such casese in C++20 before this change. Eg:

  .. code-block:: cpp

    struct P {};
    template<class S> bool operator==(const P&, const S&);

    struct A : public P {};
    struct B : public P {};

    // This equality is now ambiguous in C++20.
    bool ambiguous(A a, B b) { return a == b; }

    template<class S> bool operator!=(const P&, const S&);
    // Ok. Found a matching operator!=.
    bool fine(A a, B b) { return a == b; }

  To reduce such widespread breakages, as an extension, Clang accepts this code
  with an existing warning ``-Wambiguous-reversed-operator`` warning.
  Fixes `GH <https://github.com/llvm/llvm-project/issues/53954>`_.

- The CMake variable ``GCC_INSTALL_PREFIX`` (which sets the default
  ``--gcc-toolchain=``) is deprecated and will be removed. Specify
  ``--gcc-install-dir=`` or ``--gcc-triple=`` in a `configuration file
  <https://clang.llvm.org/docs/UsersManual.html#configuration-files>` as a
  replacement.
  (`#77537 <https://github.com/llvm/llvm-project/pull/77537>`_)

C/C++ Language Potentially Breaking Changes
-------------------------------------------

- The default extension name for PCH generation (``-c -xc-header`` and ``-c
  -xc++-header``) is now ``.pch`` instead of ``.gch``.
- ``-include a.h`` probing ``a.h.gch`` will now ignore ``a.h.gch`` if it is not
  a clang pch file or a directory containing any clang pch file.
- Fixed a bug that caused ``__has_cpp_attribute`` and ``__has_c_attribute``
  return incorrect values for some C++-11-style attributes. Below is a complete
  list of behavior changes.

  .. csv-table::
    :header: Test, Old value, New value

    ``__has_cpp_attribute(unused)``,                    201603, 0
    ``__has_cpp_attribute(gnu::unused)``,               201603, 1
    ``__has_c_attribute(unused)``,                      202106, 0
    ``__has_cpp_attribute(clang::fallthrough)``,        201603, 1
    ``__has_cpp_attribute(gnu::fallthrough)``,          201603, 1
    ``__has_c_attribute(gnu::fallthrough)``,            201910, 1
    ``__has_cpp_attribute(warn_unused_result)``,        201907, 0
    ``__has_cpp_attribute(clang::warn_unused_result)``, 201907, 1
    ``__has_cpp_attribute(gnu::warn_unused_result)``,   201907, 1
    ``__has_c_attribute(warn_unused_result)``,          202003, 0
    ``__has_c_attribute(gnu::warn_unused_result)``,     202003, 1

- Fixed a bug in finding matching `operator!=` while adding reversed `operator==` as
  outlined in "The Equality Operator You Are Looking For" (`P2468 <http://wg21.link/p2468r2>`_).
  Fixes (`#68901: <https://github.com/llvm/llvm-project/issues/68901>`_).

- Remove the hardcoded path to the imported modules for C++20 named modules. Now we
  require all the dependent modules to specified from the command line.
  See (`#62707: <https://github.com/llvm/llvm-project/issues/62707>`_).

C++ Specific Potentially Breaking Changes
-----------------------------------------
- The name mangling rules for function templates has been changed to take into
  account the possibility that functions could be overloaded on their template
  parameter lists or requires-clauses. This causes mangled names to change for
  function templates in the following cases:

  - When a template parameter in a function template depends on a previous
    template parameter, such as ``template<typename T, T V> void f()``.
  - When the function has any constraints, whether from constrained template
      parameters or requires-clauses.
  - When the template parameter list includes a deduced type -- either
      ``auto``, ``decltype(auto)``, or a deduced class template specialization
      type.
  - When a template template parameter is given a template template argument
      that has a different template parameter list.

  This fixes a number of issues where valid programs would be rejected due to
  mangling collisions, or would in some cases be silently miscompiled. Clang
  will use the old manglings if ``-fclang-abi-compat=17`` or lower is
  specified.
  (`#48216 <https://github.com/llvm/llvm-project/issues/48216>`_),
  (`#49884 <https://github.com/llvm/llvm-project/issues/49884>`_), and
  (`#61273 <https://github.com/llvm/llvm-project/issues/61273>`_)

- The `ClassScopeFunctionSpecializationDecl` AST node has been removed.
  Dependent class scope explicit function template specializations now use
  `DependentFunctionTemplateSpecializationInfo` to store candidate primary
  templates and explicit template arguments. This should not impact users of
  Clang as a compiler, but it may break assumptions in Clang-based tools
  iterating over the AST.

- The warning `-Wenum-constexpr-conversion` is now also enabled by default on
  system headers and macros. It will be turned into a hard (non-downgradable)
  error in the next Clang release.

- The flag `-fdelayed-template-parsing` won't be enabled by default with C++20
  when targetting MSVC to match the behavior of MSVC.
  (`MSVC Docs <https://learn.microsoft.com/en-us/cpp/build/reference/permissive-standards-conformance?view=msvc-170>`_)

ABI Changes in This Version
---------------------------
- Following the SystemV ABI for x86-64, ``__int128`` arguments will no longer
  be split between a register and a stack slot.

AST Dumping Potentially Breaking Changes
----------------------------------------
- When dumping a sugared type, Clang will no longer print the desugared type if
  its textual representation is the same as the sugared one. This applies to
  both text dumps of the form ``'foo':'foo'`` which will now be dumped as just
  ``'foo'``, and JSON dumps of the form:

  .. code-block:: json

    "type": {
      "qualType": "foo",
      "desugaredQualType": "foo"
    }

  which will now be dumped as just:

  .. code-block:: json

    "type": {
      "qualType": "foo"
    }

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
- Implemented `P0847R7: Deducing this <https://wg21.link/P0847R7>`_. Some related core issues were also
  implemented (`CWG2553 <https://wg21.link/CWG2553>`_, `CWG2554 <https://wg21.link/CWG2554>`_,
  `CWG2653 <https://wg21.link/CWG2653>`_, `CWG2687 <https://wg21.link/CWG2687>`_). Because the
  support for this feature is still experimental, the feature test macro ``__cpp_explicit_this_parameter``
  was not set in this version.

- Added a separate warning to warn the use of attributes on lambdas as a C++23 extension
  in previous language versions: ``-Wc++23-lambda-attributes``.

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

- Implemented `P2864R2 Remove Deprecated Arithmetic Conversion on Enumerations From C++26 <https://wg21.link/P2864R2>`_.

- Implemented `P2361R6 Template parameter initialization <https://wg21.link/P2308R1>`_.
  This change is applied as a DR in all language modes.


Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Implemented `CWG2598 <https://wg21.link/CWG2598>`_ and `CWG2096 <https://wg21.link/CWG2096>`_,
  making unions (that have either no members or at least one literal member) literal types.
  (`#77924: <https://github.com/llvm/llvm-project/issues/77924>`_).


C Language Changes
------------------
- ``structs``, ``unions``, and ``arrays`` that are const may now be used as
  constant expressions.  This change is more consistent with the behavior of
  GCC.
- Enums will now be represented in TBAA metadata using their actual underlying
  integer type. Previously they were treated as chars, which meant they could
  alias with all other types.
- Clang now supports the C-only attribute ``counted_by``. When applied to a
  struct's flexible array member, it points to the struct field that holds the
  number of elements in the flexible array member. This information can improve
  the results of the array bound sanitizer and the
  ``__builtin_dynamic_object_size`` builtin.

C23 Feature Support
^^^^^^^^^^^^^^^^^^^
- Clang now accepts ``-std=c23`` and ``-std=gnu23`` as language standard modes,
  and the ``__STDC_VERSION__`` macro now expands to ``202311L`` instead of its
  previous placeholder value. Clang continues to accept ``-std=c2x`` and
  ``-std=gnu2x`` as aliases for C23 and GNU C23, respectively.
- Clang now supports `requires c23` for module maps.
- Clang now supports ``N3007 Type inference for object definitions``.

- Clang now supports ``<stdckdint.h>`` which defines several macros for performing
  checked integer arithmetic. It is also exposed in pre-C23 modes.

- Completed the implementation of
  `N2508 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2508.pdf>`_. We
  previously implemented allowing a label at the end of a compound statement,
  and now we've implemented allowing a label to be followed by a declaration
  instead of a statement.

Non-comprehensive list of changes in this release
-------------------------------------------------

* Clang now has a ``__builtin_vectorelements()`` function that determines the number of elements in a vector.
  For fixed-sized vectors, e.g., defined via ``__attribute__((vector_size(N)))`` or ARM NEON's vector types
  (e.g., ``uint16x8_t``), this returns the constant number of elements at compile-time.
  For scalable vectors, e.g., SVE or RISC-V V, the number of elements is not known at compile-time and is
  determined at runtime.
* The ``__datasizeof`` keyword has been added. It is similar to ``sizeof``
  except that it returns the size of a type ignoring tail padding.
* ``__builtin_classify_type()`` now classifies ``_BitInt`` values as the return value ``18``
  and vector types as return value ``19``, to match GCC 14's behavior.
* The default value of `_MSC_VER` was raised from 1920 to 1933.
* Since MSVC 19.33 added undocumented attribute ``[[msvc::constexpr]]``, this release adds the attribute as well.

* Added ``#pragma clang fp reciprocal``.

* The version of Unicode used by Clang (primarily to parse identifiers) has been updated to 15.1.

New Compiler Flags
------------------

* ``-fverify-intermediate-code`` and its complement ``-fno-verify-intermediate-code``.
  Enables or disables verification of the generated LLVM IR.
  Users can pass this to turn on extra verification to catch certain types of
  compiler bugs at the cost of extra compile time.
  Since enabling the verifier adds a non-trivial cost of a few percent impact on
  build times, it's disabled by default, unless your LLVM distribution itself is
  compiled with runtime checks enabled.
* ``-fkeep-system-includes`` modifies the behavior of the ``-E`` option,
  preserving ``#include`` directives for "system" headers instead of copying
  the preprocessed text to the output. This can greatly reduce the size of the
  preprocessed output, which can be helpful when trying to reduce a test case.
* ``-fassume-nothrow-exception-dtor`` is added to assume that the destructor of
  a thrown exception object will not throw. The generated code for catch
  handlers will be smaller. A throw expression of a type with a
  potentially-throwing destructor will lead to an error.

* ``-fopenacc`` was added as a part of the effort to support OpenACC in clang.

* ``-fcx-limited-range`` enables the naive mathematical formulas for complex
  division and multiplication with no NaN checking of results. The default is
  ``-fno-cx-limited-range``, but this option is enabled by ``-ffast-math``.

* ``-fcx-fortran-rules`` enables the naive mathematical formulas for complex
  multiplication and enables application of Smith's algorithm for complex
  division. See SMITH, R. L. Algorithm 116: Complex division. Commun. ACM 5, 8
  (1962). The default is ``-fno-cx-fortran-rules``.


Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

* ``-Woverriding-t-option`` is renamed to ``-Woverriding-option``.
* ``-Winterrupt-service-routine`` is renamed to ``-Wexcessive-regsave`` as a generalization
* ``-frewrite-includes`` now guards the original #include directives with
  ``__CLANG_REWRITTEN_INCLUDES``, and ``__CLANG_REWRITTEN_SYSTEM_INCLUDES`` as
  appropriate.
* Introducing a new default calling convention for ``-fdefault-calling-conv``:
  ``rtdcall``. This new default CC only works for M68k and will use the new
  ``m68k_rtdcc`` CC on every functions that are not variadic. The ``-mrtd``
  driver/frontend flag has the same effect when targeting M68k.

Removed Compiler Flags
-------------------------

* ``-enable-trivial-auto-var-init-zero-knowing-it-will-be-removed-from-clang`` has been removed.
  It has not been needed to enable ``-ftrivial-auto-var-init=zero`` since Clang 16.

Attribute Changes in Clang
--------------------------
- On X86, a warning is now emitted if a function with ``__attribute__((no_caller_saved_registers))``
  calls a function without ``__attribute__((no_caller_saved_registers))``, and is not compiled with
  ``-mgeneral-regs-only``
- On X86, a function with ``__attribute__((interrupt))`` can now call a function without
  ``__attribute__((no_caller_saved_registers))`` provided that it is compiled with ``-mgeneral-regs-only``

- When a non-variadic function is decorated with the ``format`` attribute,
  Clang now checks that the format string would match the function's parameters'
  types after default argument promotion. As a result, it's no longer an
  automatic diagnostic to use parameters of types that the format style
  supports but that are never the result of default argument promotion, such as
  ``float``. (`#59824: <https://github.com/llvm/llvm-project/issues/59824>`_)

- Clang now supports ``[[clang::preferred_type(type-name)]]`` as an attribute
  which can be applied to a bit-field. This attribute helps to map a bit-field
  back to a particular type that may be better-suited to representing the bit-
  field but cannot be used for other reasons and will impact the debug
  information generated for the bit-field. This is most useful when mapping a
  bit-field of basic integer type back to a ``bool`` or an enumeration type,
  e.g.,

  .. code-block:: c++

      enum E { Apple, Orange, Pear };
      struct S {
        [[clang::preferred_type(E)]] unsigned FruitKind : 2;
      };

  When viewing ``S::FruitKind`` in a debugger, it will behave as if the member
  was declared as type ``E`` rather than ``unsigned``.

- Clang now warns you that the ``_Alignas`` attribute on declaration specifiers
  is ignored, changed from the former incorrect suggestion to move it past
  declaration specifiers. (`#58637 <https://github.com/llvm/llvm-project/issues/58637>`_)

- Clang now introduced ``[[clang::coro_only_destroy_when_complete]]`` attribute
  to reduce the size of the destroy functions for coroutines which are known to
  be destroyed after having reached the final suspend point.

- Clang now introduced ``[[clang::coro_return_type]]`` and ``[[clang::coro_wrapper]]``
  attributes. A function returning a type marked with ``[[clang::coro_return_type]]``
  should be a coroutine. A non-coroutine function marked with ``[[clang::coro_wrapper]]``
  is still allowed to return the such a type. This is helpful for analyzers to recognize coroutines from the function signatures.

- Clang now supports ``[[clang::code_align(N)]]`` as an attribute which can be
  applied to a loop and specifies the byte alignment for a loop. This attribute
  accepts a positive integer constant initialization expression indicating the
  number of bytes for the minimum alignment boundary. Its value must be a power
  of 2, between 1 and 4096(inclusive).

  .. code-block:: c++

      void Array(int *array, size_t n) {
        [[clang::code_align(64)]] for (int i = 0; i < n; ++i) array[i] = 0;
      }

      template<int A>
      void func() {
        [[clang::code_align(A)]] for(;;) { }
      }

- Clang now introduced ``[[clang::coro_lifetimebound]]`` attribute.
  All parameters of a function are considered to be lifetime bound if the function
  returns a type annotated with ``[[clang::coro_lifetimebound]]`` and ``[[clang::coro_return_type]]``.
  This analysis can be disabled for a function by annotating the function with ``[[clang::coro_disable_lifetimebound]]``.

Improvements to Clang's diagnostics
-----------------------------------
- Clang constexpr evaluator now prints template arguments when displaying
  template-specialization function calls.
- Clang contexpr evaluator now displays notes as well as an error when a constructor
  of a base class is not called in the constructor of its derived class.
- Clang no longer emits ``-Wmissing-variable-declarations`` for variables declared
  with the ``register`` storage class.
- Clang's ``-Wswitch-default`` flag now diagnoses whenever a ``switch`` statement
  does not have a ``default`` label.
- Clang's ``-Wtautological-negation-compare`` flag now diagnoses logical
  tautologies like ``x && !x`` and ``!x || x`` in expressions. This also
  makes ``-Winfinite-recursion`` diagnose more cases.
  (`#56035: <https://github.com/llvm/llvm-project/issues/56035>`_).
- Clang constexpr evaluator now diagnoses compound assignment operators against
  uninitialized variables as a read of uninitialized object.
  (`#51536 <https://github.com/llvm/llvm-project/issues/51536>`_)
- Clang's ``-Wformat-truncation`` now diagnoses ``snprintf`` call that is known to
  result in string truncation.
  (`#64871: <https://github.com/llvm/llvm-project/issues/64871>`_).
  Existing warnings that similarly warn about the overflow in ``sprintf``
  now falls under its own warning group ```-Wformat-overflow`` so that it can
  be disabled separately from ``Wfortify-source``.
  These two new warning groups have subgroups ``-Wformat-truncation-non-kprintf``
  and ``-Wformat-overflow-non-kprintf``, respectively. These subgroups are used when
  the format string contains ``%p`` format specifier.
  Because Linux kernel's codebase has format extensions for ``%p``, kernel developers
  are encouraged to disable these two subgroups by setting ``-Wno-format-truncation-non-kprintf``
  and ``-Wno-format-overflow-non-kprintf`` in order to avoid false positives on
  the kernel codebase.
  Also clang no longer emits false positive warnings about the output length of
  ``%g`` format specifier and about ``%o, %x, %X`` with ``#`` flag.
- Clang now emits ``-Wcast-qual`` for functional-style cast expressions.
- Clang no longer emits irrelevant notes about unsatisfied constraint expressions
  on the left-hand side of ``||`` when the right-hand side constraint is satisfied.
  (`#54678: <https://github.com/llvm/llvm-project/issues/54678>`_).
- Clang now prints its 'note' diagnostic in cyan instead of black, to be more compatible
  with terminals with dark background colors. This is also more consistent with GCC.
- Clang now displays an improved diagnostic and a note when a defaulted special
  member is marked ``constexpr`` in a class with a virtual base class
  (`#64843: <https://github.com/llvm/llvm-project/issues/64843>`_).
- ``-Wfixed-enum-extension`` and ``-Wmicrosoft-fixed-enum`` diagnostics are no longer
  emitted when building as C23, since C23 standardizes support for enums with a
  fixed underlying type.
- When describing the failure of static assertion of `==` expression, clang prints the integer
  representation of the value as well as its character representation when
  the user-provided expression is of character type. If the character is
  non-printable, clang now shows the escpaed character.
  Clang also prints multi-byte characters if the user-provided expression
  is of multi-byte character type.

  *Example Code*:

  .. code-block:: c++

     static_assert("A\n"[1] == U'🌍');

  *BEFORE*:

  .. code-block:: text

    source:1:15: error: static assertion failed due to requirement '"A\n"[1] == U'\U0001f30d''
    1 | static_assert("A\n"[1] == U'🌍');
      |               ^~~~~~~~~~~~~~~~~
    source:1:24: note: expression evaluates to ''
    ' == 127757'
    1 | static_assert("A\n"[1] == U'🌍');
      |               ~~~~~~~~~^~~~~~~~

  *AFTER*:

  .. code-block:: text

    source:1:15: error: static assertion failed due to requirement '"A\n"[1] == U'\U0001f30d''
    1 | static_assert("A\n"[1] == U'🌍');
      |               ^~~~~~~~~~~~~~~~~
    source:1:24: note: expression evaluates to ''\n' (0x0A, 10) == U'🌍' (0x1F30D, 127757)'
    1 | static_assert("A\n"[1] == U'🌍');
      |               ~~~~~~~~~^~~~~~~~
- Clang now always diagnoses when using non-standard layout types in ``offsetof`` .
  (`#64619: <https://github.com/llvm/llvm-project/issues/64619>`_)
- Clang now diagnoses redefined defaulted constructor when redefined
  defaulted constructor with different exception specs.
  (`#69094: <https://github.com/llvm/llvm-project/issues/69094>`_)
- Clang now diagnoses use of variable-length arrays in C++ by default (and
  under ``-Wall`` in GNU++ mode). This is an extension supported by Clang and
  GCC, but is very easy to accidentally use without realizing it's a
  nonportable construct that has different semantics from a constant-sized
  array. (`#62836 <https://github.com/llvm/llvm-project/issues/62836>`_)

- Clang changed the order in which it displays candidate functions on overloading failures.
  Previously, Clang used definition of ordering from the C++ Standard. The order defined in
  the Standard is partial and is not suited for sorting. Instead, Clang now uses a strict
  order that still attempts to push more relevant functions to the top by comparing their
  corresponding conversions. In some cases, this results in better order. E.g., for the
  following code

  .. code-block:: cpp

      struct Foo {
        operator int();
        operator const char*();
      };

      void test() { Foo() - Foo(); }

  Clang now produces a list with two most relevant builtin operators at the top,
  i.e. ``operator-(int, int)`` and ``operator-(const char*, const char*)``.
  Previously ``operator-(const char*, const char*)`` was the first element,
  but ``operator-(int, int)`` was only the 13th element in the output.
  However, new implementation does not take into account some aspects of
  C++ semantics, e.g. which function template is more specialized. This
  can sometimes lead to worse ordering.


- When describing a warning/error in a function-style type conversion Clang underlines only until
  the end of the expression we convert from. Now Clang underlines until the closing parenthesis.

  Before:

  .. code-block:: text

    warning: cast from 'long (*)(const int &)' to 'decltype(fun_ptr)' (aka 'long (*)(int &)') converts to incompatible function type [-Wcast-function-type-strict]
    24 | return decltype(fun_ptr)( f_ptr /*comment*/);
       |        ^~~~~~~~~~~~~~~~~~~~~~~~

  After:

  .. code-block:: text

    warning: cast from 'long (*)(const int &)' to 'decltype(fun_ptr)' (aka 'long (*)(int &)') converts to incompatible function type [-Wcast-function-type-strict]
    24 | return decltype(fun_ptr)( f_ptr /*comment*/);
       |        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``-Wzero-as-null-pointer-constant`` diagnostic is no longer emitted when using ``__null``
  (or, more commonly, ``NULL`` when the platform defines it as ``__null``) to be more consistent
  with GCC.
- Clang will warn on deprecated specializations used in system headers when their instantiation
  is caused by user code.
- Clang will now print ``static_assert`` failure details for arithmetic binary operators.
  Example:

  .. code-block:: cpp

    static_assert(1 << 4 == 15);

  will now print:

  .. code-block:: text

    error: static assertion failed due to requirement '1 << 4 == 15'
       48 | static_assert(1 << 4 == 15);
          |               ^~~~~~~~~~~~
    note: expression evaluates to '16 == 15'
       48 | static_assert(1 << 4 == 15);
          |               ~~~~~~~^~~~~

- Clang now diagnoses definitions of friend function specializations, e.g. ``friend void f<>(int) {}``.
- Clang now diagnoses narrowing conversions involving const references.
  (`#63151: <https://github.com/llvm/llvm-project/issues/63151>`_).
- Clang now diagnoses unexpanded packs within the template argument lists of function template specializations.
- Clang now diagnoses attempts to bind a bitfield to an NTTP of a reference type as erroneous
  converted constant expression and not as a reference to subobject.
- Clang now diagnoses the requirement that non-template friend declarations with requires clauses
  and template friend declarations with a constraint that depends on a template parameter from an
  enclosing template must be a definition.
- Clang now diagnoses function/variable templates that shadow their own template parameters, e.g. ``template<class T> void T();``.

- Clang now emits more descriptive diagnostics for 'unusual' expressions (e.g. incomplete index
  expressions on matrix types or builtin functions without an argument list) as placement-args
  to new-expressions.

  Before:

  .. code-block:: text

    error: no matching function for call to 'operator new'
       13 |     new (__builtin_memset) S {};
          |     ^   ~~~~~~~~~~~~~~~~~~

    note: candidate function not viable: no known conversion from '<builtin fn type>' to 'int' for 2nd argument
        5 |     void* operator new(__SIZE_TYPE__, int);
          |           ^

  After:

  .. code-block:: text

    error: builtin functions must be directly called
       13 |     new (__builtin_memset) S {};
          |          ^


Improvements to Clang's time-trace
----------------------------------
- Two time-trace scope variables are added. A time trace scope variable of
  ``ParseDeclarationOrFunctionDefinition`` with the function's source location
  is added to record the time spent parsing the function's declaration or
  definition. Another time trace scope variable of ``ParseFunctionDefinition``
  is also added to record the name of the defined function.

Bug Fixes in This Version
-------------------------
- Fixed an issue where a class template specialization whose declaration is
  instantiated in one module and whose definition is instantiated in another
  module may end up with members associated with the wrong declaration of the
  class, which can result in miscompiles in some cases.
- Fix crash on use of a variadic overloaded operator.
  (`#42535 <https://github.com/llvm/llvm-project/issues/42535>`_)
- Fix a hang on valid C code passing a function type as an argument to
  ``typeof`` to form a function declaration.
  (`#64713 <https://github.com/llvm/llvm-project/issues/64713>`_)
- Clang now reports missing-field-initializers warning for missing designated
  initializers in C++.
  (`#56628 <https://github.com/llvm/llvm-project/issues/56628>`_)
- Clang now respects ``-fwrapv`` and ``-ftrapv`` for ``__builtin_abs`` and
  ``abs`` builtins.
  (`#45129 <https://github.com/llvm/llvm-project/issues/45129>`_,
  `#45794 <https://github.com/llvm/llvm-project/issues/45794>`_)
- Fixed an issue where accesses to the local variables of a coroutine during
  ``await_suspend`` could be misoptimized, including accesses to the awaiter
  object itself.
  (`#56301 <https://github.com/llvm/llvm-project/issues/56301>`_)
  The current solution may bring performance regressions if the awaiters have
  non-static data members. See
  `#64945 <https://github.com/llvm/llvm-project/issues/64945>`_ for details.
- Clang now prints unnamed members in diagnostic messages instead of giving an
  empty ''. Fixes
  (`#63759 <https://github.com/llvm/llvm-project/issues/63759>`_)
- Fix crash in __builtin_strncmp and related builtins when the size value
  exceeded the maximum value representable by int64_t. Fixes
  (`#64876 <https://github.com/llvm/llvm-project/issues/64876>`_)
- Fixed an assertion if a function has cleanups and fatal erors.
  (`#48974 <https://github.com/llvm/llvm-project/issues/48974>`_)
- Clang now emits an error if it is not possible to deduce array size for a
  variable with incomplete array type.
  (`#37257 <https://github.com/llvm/llvm-project/issues/37257>`_)
- Clang's ``-Wunused-private-field`` no longer warns on fields whose type is
  declared with ``[[maybe_unused]]``.
  (`#61334 <https://github.com/llvm/llvm-project/issues/61334>`_)
- For function multi-versioning using the ``target``, ``target_clones``, or
  ``target_version`` attributes, remove comdat for internal linkage functions.
  (`#65114 <https://github.com/llvm/llvm-project/issues/65114>`_)
- Clang now reports ``-Wformat`` for bool value and char specifier confusion
  in scanf. Fixes
  (`#64987 <https://github.com/llvm/llvm-project/issues/64987>`_)
- Support MSVC predefined macro expressions in constant expressions and in
  local structs.
- Correctly parse non-ascii identifiers that appear immediately after a line splicing
  (`#65156 <https://github.com/llvm/llvm-project/issues/65156>`_)
- Clang no longer considers the loss of ``__unaligned`` qualifier from objects as
  an invalid conversion during method function overload resolution.
- Fix lack of comparison of declRefExpr in ASTStructuralEquivalence
  (`#66047 <https://github.com/llvm/llvm-project/issues/66047>`_)
- Fix parser crash when dealing with ill-formed objective C++ header code. Fixes
  (`#64836 <https://github.com/llvm/llvm-project/issues/64836>`_)
- Fix crash in implicit conversions from initialize list to arrays of unknown
  bound for C++20. Fixes
  (`#62945 <https://github.com/llvm/llvm-project/issues/62945>`_)
- Clang now allows an ``_Atomic`` qualified integer in a switch statement. Fixes
  (`#65557 <https://github.com/llvm/llvm-project/issues/65557>`_)
- Fixes crash when trying to obtain the common sugared type of
  `decltype(instantiation-dependent-expr)`.
  Fixes (`#67603 <https://github.com/llvm/llvm-project/issues/67603>`_)
- Fixes a crash caused by a multidimensional array being captured by a lambda
  (`#67722 <https://github.com/llvm/llvm-project/issues/67722>`_).
- Fixes a crash when instantiating a lambda with requires clause.
  (`#64462 <https://github.com/llvm/llvm-project/issues/64462>`_)
- Fixes a regression where the ``UserDefinedLiteral`` was not properly preserved
  while evaluating consteval functions. (`#63898 <https://github.com/llvm/llvm-project/issues/63898>`_).
- Fix a crash when evaluating value-dependent structured binding
  variables at compile time.
  Fixes (`#67690 <https://github.com/llvm/llvm-project/issues/67690>`_)
- Fixes a ``clang-17`` regression where ``LLVM_UNREACHABLE_OPTIMIZE=OFF``
  cannot be used with ``Release`` mode builds. (`#68237 <https://github.com/llvm/llvm-project/issues/68237>`_).
- Fix crash in evaluating ``constexpr`` value for invalid template function.
  Fixes (`#68542 <https://github.com/llvm/llvm-project/issues/68542>`_)
- Clang will correctly evaluate ``noexcept`` expression for template functions
  of template classes. Fixes
  (`#68543 <https://github.com/llvm/llvm-project/issues/68543>`_,
  `#42496 <https://github.com/llvm/llvm-project/issues/42496>`_,
  `#77071 <https://github.com/llvm/llvm-project/issues/77071>`_,
  `#77411 <https://github.com/llvm/llvm-project/issues/77411>`_)
- Fixed an issue when a shift count larger than ``__INT64_MAX__``, in a right
  shift operation, could result in missing warnings about
  ``shift count >= width of type`` or internal compiler error.
- Fixed an issue with computing the common type for the LHS and RHS of a `?:`
  operator in C. No longer issuing a confusing diagnostic along the lines of
  "incompatible operand types ('foo' and 'foo')" with extensions such as matrix
  types. Fixes (`#69008 <https://github.com/llvm/llvm-project/issues/69008>`_)
- Clang no longer permits using the `_BitInt` types as an underlying type for an
  enumeration as specified in the C23 Standard.
  Fixes (`#69619 <https://github.com/llvm/llvm-project/issues/69619>`_)
- Fixed an issue when a shift count specified by a small constant ``_BitInt()``,
  in a left shift operation, could result in a faulty warnings about
  ``shift count >= width of type``.
- Clang now accepts anonymous members initialized with designated initializers
  inside templates.
  Fixes (`#65143 <https://github.com/llvm/llvm-project/issues/65143>`_)
- Fix crash in formatting the real/imaginary part of a complex lvalue.
  Fixes (`#69218 <https://github.com/llvm/llvm-project/issues/69218>`_)
- No longer use C++ ``thread_local`` semantics in C23 when using
  ``thread_local`` instead of ``_Thread_local``.
  Fixes (`#70068 <https://github.com/llvm/llvm-project/issues/70068>`_) and
  (`#69167 <https://github.com/llvm/llvm-project/issues/69167>`_)
- Fix crash in evaluating invalid lambda expression which forget capture this.
  Fixes (`#67687 <https://github.com/llvm/llvm-project/issues/67687>`_)
- Fix crash from constexpr evaluator evaluating uninitialized arrays as rvalue.
  Fixes (`#67317 <https://github.com/llvm/llvm-project/issues/67317>`_)
- Clang now properly diagnoses use of stand-alone OpenMP directives after a
  label (including ``case`` or ``default`` labels).
- Fix compiler memory leak for enums with underlying type larger than 64 bits.
  Fixes (`#78311 <https://github.com/llvm/llvm-project/pull/78311>`_)

  Before:

  .. code-block:: c++

    label:
    #pragma omp barrier // ok

  After:

  .. code-block:: c++

    label:
    #pragma omp barrier // error: '#pragma omp barrier' cannot be an immediate substatement

- Fixed an issue that a benign assertion might hit when instantiating a pack expansion
  inside a lambda. (`#61460 <https://github.com/llvm/llvm-project/issues/61460>`_)
- Fix crash during instantiation of some class template specializations within class
  templates. Fixes (`#70375 <https://github.com/llvm/llvm-project/issues/70375>`_)
- Fix crash during code generation of C++ coroutine initial suspend when the return
  type of await_resume is not trivially destructible.
  Fixes (`#63803 <https://github.com/llvm/llvm-project/issues/63803>`_)
- ``__is_trivially_relocatable`` no longer returns true for non-object types
  such as references and functions.
  Fixes (`#67498 <https://github.com/llvm/llvm-project/issues/67498>`_)
- Fix crash when the object used as a ``static_assert`` message has ``size`` or ``data`` members
  which are not member functions.
- Support UDLs in ``static_assert`` message.
- Fixed false positive error emitted by clang when performing qualified name
  lookup and the current class instantiation has dependent bases.
  Fixes (`#13826 <https://github.com/llvm/llvm-project/issues/13826>`_)
- Fix a ``clang-17`` regression where a templated friend with constraints is not
  properly applied when its parameters reference an enclosing non-template class.
  Fixes (`#71595 <https://github.com/llvm/llvm-project/issues/71595>`_)
- Fix the name of the ifunc symbol emitted for multiversion functions declared with the
  ``target_clones`` attribute. This addresses a linker error that would otherwise occur
  when these functions are referenced from other TUs.
- Fixes compile error that double colon operator cannot resolve macro with parentheses.
  Fixes (`#64467 <https://github.com/llvm/llvm-project/issues/64467>`_)
- Clang's ``-Wchar-subscripts`` no longer warns on chars whose values are known non-negative constants.
  Fixes (`#18763 <https://github.com/llvm/llvm-project/issues/18763>`_)
- Fix crash due to incorrectly allowing conversion functions in copy elision.
  Fixes (`#39319 <https://github.com/llvm/llvm-project/issues/39319>`_) and
  (`#60182 <https://github.com/llvm/llvm-project/issues/60182>`_) and
  (`#62157 <https://github.com/llvm/llvm-project/issues/62157>`_) and
  (`#64885 <https://github.com/llvm/llvm-project/issues/64885>`_) and
  (`#65568 <https://github.com/llvm/llvm-project/issues/65568>`_)
- Fix an issue where clang doesn't respect detault template arguments that
  are added in a later redeclaration for CTAD.
  Fixes (`#69987 <https://github.com/llvm/llvm-project/issues/69987>`_)
- Fix an issue where CTAD fails for explicit type conversion.
  Fixes (`#64347 <https://github.com/llvm/llvm-project/issues/64347>`_)
- Fix crash when using C++ only tokens like ``::`` in C compiler clang.
  Fixes (`#73559 <https://github.com/llvm/llvm-project/issues/73559>`_)
- Clang now accepts recursive non-dependent calls to functions with deduced
  return type.
  Fixes (`#71015 <https://github.com/llvm/llvm-project/issues/71015>`_)
- Fix assertion failure when initializing union containing struct with
  flexible array member using empty initializer list.
  Fixes (`#77085 <https://github.com/llvm/llvm-project/issues/77085>`_)
- Fix assertion crash due to failed scope restoring caused by too-early VarDecl
  invalidation by invalid initializer Expr.
  Fixes (`#30908 <https://github.com/llvm/llvm-project/issues/30908>`_)
- Clang now emits correct source location for code-coverage regions in `if constexpr`
  and `if consteval` branches.
  Fixes (`#54419 <https://github.com/llvm/llvm-project/issues/54419>`_)
- Fix assertion failure when declaring a template friend function with
  a constrained parameter in a template class that declares a class method
  or lambda at different depth.
  Fixes (`#75426 <https://github.com/llvm/llvm-project/issues/75426>`_)
- Fix an issue where clang cannot find conversion function with template
  parameter when instantiation of template class.
  Fixes (`#77583 <https://github.com/llvm/llvm-project/issues/77583>`_)
- Fix an issue where CTAD fails for function-type/array-type arguments.
  Fixes (`#51710 <https://github.com/llvm/llvm-project/issues/51710>`_)
- Fix crashes when using the binding decl from an invalid structured binding.
  Fixes (`#67495 <https://github.com/llvm/llvm-project/issues/67495>`_) and
  (`#72198 <https://github.com/llvm/llvm-project/issues/72198>`_)
- Fix assertion failure when call noreturn-attribute function with musttail
  attribute.
  Fixes (`#76631 <https://github.com/llvm/llvm-project/issues/76631>`_)
  - The MS ``__noop`` builtin without an argument list is now accepted
  in the placement-args of new-expressions, matching MSVC's behaviour.
- Fix an issue that caused MS ``__decspec(property)`` accesses as well as
  Objective-C++ property accesses to not be converted to a function call
  to the getter in the placement-args of new-expressions.
  Fixes (`#65053 <https://github.com/llvm/llvm-project/issues/65053>`_)

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
  (`#64138 <https://github.com/llvm/llvm-project/issues/64138>`_) and
  (`#71684 <https://github.com/llvm/llvm-project/issues/71684>`_).

- Update ``FunctionDeclBitfields.NumFunctionDeclBits``. This fixes:
  (`#64171 <https://github.com/llvm/llvm-project/issues/64171>`_).

- Expressions producing ``nullptr`` are correctly evaluated
  by the constant interpreter when appearing as the operand
  of a binary comparison.
  (`#64923 <https://github.com/llvm/llvm-project/issues/64923>`_)

- Fix a crash when an immediate invocation is not a constant expression
  and appear in an implicit cast.
  (`#64949 <https://github.com/llvm/llvm-project/issues/64949>`_).

- Fix crash when parsing ill-formed lambda trailing return type. Fixes:
  (`#64962 <https://github.com/llvm/llvm-project/issues/64962>`_) and
  (`#28679 <https://github.com/llvm/llvm-project/issues/28679>`_).

- Fix a crash caused by substitution failure in expression requirements.
  (`#64172 <https://github.com/llvm/llvm-project/issues/64172>`_) and
  (`#64723 <https://github.com/llvm/llvm-project/issues/64723>`_).

- Fix crash when parsing the requires clause of some generic lambdas.
  (`#64689 <https://github.com/llvm/llvm-project/issues/64689>`_)

- Fix crash when the trailing return type of a generic and dependent
  lambda refers to an init-capture.
  (`#65067 <https://github.com/llvm/llvm-project/issues/65067>`_ and
  `#63675 <https://github.com/llvm/llvm-project/issues/63675>`_)

- Clang now properly handles out of line template specializations when there is
  a non-template inner-class between the function and the class template.
  (`#65810 <https://github.com/llvm/llvm-project/issues/65810>`_)

- Fix a crash when calling a non-constant immediate function
  in the initializer of a static data member.
  (`#65985 <https://github.com/llvm/llvm-project/issues/65985>_`).
- Clang now properly converts static lambda call operator to function
  pointers on win32.
  (`#62594 <https://github.com/llvm/llvm-project/issues/62594>`_)

- Fixed some cases where the source location for an instantiated specialization
  of a function template or a member function of a class template was assigned
  the location of a non-defining declaration rather than the location of the
  definition the specialization was instantiated from.
  (`#26057 <https://github.com/llvm/llvm-project/issues/26057>`_`)

- Fix a crash when a default member initializer of a base aggregate
  makes an invalid call to an immediate function.
  (`#66324 <https://github.com/llvm/llvm-project/issues/66324>`_)

- Fix crash for a lambda attribute with a statement expression
  that contains a `return`.
  (`#48527 <https://github.com/llvm/llvm-project/issues/48527>`_)

- Clang now no longer asserts when an UnresolvedLookupExpr is used as an
  expression requirement. (`#66612 https://github.com/llvm/llvm-project/issues/66612`)

- Clang now disambiguates NTTP types when printing diagnostics where the
  NTTP types are compared with the 'diff' method.
  (`#66744 https://github.com/llvm/llvm-project/issues/66744`)

- Fix crash caused by a spaceship operator returning a comparision category by
  reference. Fixes:
  (`#64162 <https://github.com/llvm/llvm-project/issues/64162>`_)
- Fix a crash when calling a consteval function in an expression used as
  the size of an array.
  (`#65520 <https://github.com/llvm/llvm-project/issues/65520>`_)

- Clang no longer tries to capture non-odr-used variables that appear
  in the enclosing expression of a lambda expression with a noexcept specifier.
  (`#67492 <https://github.com/llvm/llvm-project/issues/67492>`_)

- Fix crash when fold expression was used in the initialization of default
  argument. Fixes:
  (`#67395 <https://github.com/llvm/llvm-project/issues/67395>`_)

- Fixed a bug causing destructors of constant-evaluated structured bindings
  initialized by array elements to be called in the wrong evaluation context.

- Fix crash where ill-formed code was being treated as a deduction guide and
  we now produce a diagnostic. Fixes:
  (`#65522 <https://github.com/llvm/llvm-project/issues/65522>`_)

- Fixed a bug where clang incorrectly considered implicitly generated deduction
  guides from a non-templated constructor and a templated constructor as ambiguous,
  rather than prefer the non-templated constructor as specified in
  [standard.group]p3.

- Fixed a crash caused by incorrect handling of dependence on variable templates
  with non-type template parameters of reference type. Fixes:
  (`#65153 <https://github.com/llvm/llvm-project/issues/65153>`_)

- Clang now properly compares constraints on an out of line class template
  declaration definition. Fixes:
  (`#61763 <https://github.com/llvm/llvm-project/issues/61763>`_)

- Fix a bug where implicit deduction guides are not correctly generated for nested template
  classes. Fixes:
  (`#46200 <https://github.com/llvm/llvm-project/issues/46200>`_)
  (`#57812 <https://github.com/llvm/llvm-project/issues/57812>`_)

- Diagnose use of a variable-length array in a coroutine. The design of
  coroutines is such that it is not possible to support VLA use. Fixes:
  (`#65858 <https://github.com/llvm/llvm-project/issues/65858>`_)

- Fix bug where we were overriding zero-initialization of class members when
  default initializing a base class in a constant expression context. Fixes:
  (`#69890 <https://github.com/llvm/llvm-project/issues/69890>`_)

- Fix crash when template class static member imported to other translation unit.
  Fixes:
  (`#68769 <https://github.com/llvm/llvm-project/issues/68769>`_)

- Clang now rejects incomplete types for ``__builtin_dump_struct``. Fixes:
  (`#63506 <https://github.com/llvm/llvm-project/issues/63506>`_)

- Fixed a crash for C++98/03 while checking an ill-formed ``_Static_assert`` expression.
  Fixes: (`#72025 <https://github.com/llvm/llvm-project/issues/72025>`_)

- Clang now defers the instantiation of explicit specifier until constraint checking
  completes (except deduction guides). Fixes:
  (`#59827 <https://github.com/llvm/llvm-project/issues/59827>`_)

- Fix crash when parsing nested requirement. Fixes:
  (`#73112 <https://github.com/llvm/llvm-project/issues/73112>`_)

- Fixed a crash caused by using return type requirement in a lambda. Fixes:
  (`#63808 <https://github.com/llvm/llvm-project/issues/63808>`_)
  (`#64607 <https://github.com/llvm/llvm-project/issues/64607>`_)
  (`#64086 <https://github.com/llvm/llvm-project/issues/64086>`_)

- Fixed a crash where we lost uninstantiated constraints on placeholder NTTP packs. Fixes:
  (`#63837 <https://github.com/llvm/llvm-project/issues/63837>`_)

- Fixed a regression where clang forgets how to substitute into constraints on template-template
  parameters. Fixes:
  (`#57410 <https://github.com/llvm/llvm-project/issues/57410>`_) and
  (`#76604 <https://github.com/llvm/llvm-project/issues/57410>`_)

- Fixes CTAD for aggregates on nested template classes. Fixes:
  (`#77599 <https://github.com/llvm/llvm-project/issues/77599>`_)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed an import failure of recursive friend class template.
  `Issue 64169 <https://github.com/llvm/llvm-project/issues/64169>`_
- Remove unnecessary RecordLayout computation when importing UnaryOperator. The
  computed RecordLayout is incorrect if fields are not completely imported and
  should not be cached.
  `Issue 64170 <https://github.com/llvm/llvm-project/issues/64170>`_
- Fixed ``hasAnyBase`` not binding nodes in its submatcher.
  (`#65421 <https://github.com/llvm/llvm-project/issues/65421>`_)
- Fixed a bug where RecursiveASTVisitor fails to visit the
  initializer of a bitfield.
  `Issue 64916 <https://github.com/llvm/llvm-project/issues/64916>`_
- Fixed a bug where range-loop-analysis checks for trivial copyability,
  rather than trivial copy-constructibility
  `Issue 47355 <https://github.com/llvm/llvm-project/issues/47355>`_
- Fixed a bug where Template Instantiation failed to handle Lambda Expressions
  with certain types of Attributes.
  (`#76521 <https://github.com/llvm/llvm-project/issues/76521>`_)

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
- Fixed a crash when an ObjC ivar has an invalid type. See
  (`#68001 <https://github.com/llvm/llvm-project/pull/68001>`_)
- Fixed a crash in C when redefined struct is another nested redefinition.
  `Issue 41302 <https://github.com/llvm/llvm-project/issues/41302>`_
- Fixed a crash when ``-ast-dump=json`` was used for code using class
  template deduction guides.
- Fixed a crash when a lambda marked as ``static`` referenced a captured
  variable in an expression.
  `Issue 74608 <https://github.com/llvm/llvm-project/issues/74608>`_
- Fixed a crash with modules and a ``constexpr`` destructor.
  `Issue 68702 <https://github.com/llvm/llvm-project/issues/68702>`_


OpenACC Specific Changes
------------------------
- OpenACC Implementation effort is beginning with semantic analysis and parsing
  of OpenACC pragmas. The ``-fopenacc`` flag was added to enable these new,
  albeit incomplete changes. The ``_OPENACC`` macro is currently defined to
  ``1``, as support is too incomplete to update to a standards-required value.
- Added ``-fexperimental-openacc-macro-override``, a command line option to
  permit overriding the ``_OPENACC`` macro to be any digit-only value specified
  by the user, which permits testing the compiler against existing OpenACC
  workloads in order to evaluate implementation progress.

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

- Added option ``-m[no-]evex512`` to disable ZMM and 64-bit mask instructions
  for AVX512 features.
- Support ISA of ``USER_MSR``.
  * Support intrinsic of ``_urdmsr``.
  * Support intrinsic of ``_uwrmsr``.
- Support ISA of ``AVX10.1``.
- ``-march=pantherlake`` and ``-march=clearwaterforest`` are now supported.
- Added ABI handling for ``__float128`` to match with GCC.
- Emit warnings for options to enable knl/knm specific ISAs: AVX512PF, AVX512ER
  and PREFETCHWT1. From next version (LLVM 19), these ISAs' intrinsic supports
  will be deprecated:
  * intrinsic series of *_exp2a23_*
  * intrinsic series of *_rsqrt28_*
  * intrinsic series of *_rcp28_*
  * intrinsic series of *_prefetch_i[3|6][2|4]gather_*
  * intrinsic series of *_prefetch_i[3|6][2|4]scatter_*

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

- C++ function name mangling has been changed to align with the specification
  (https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst).
  This affects C++ functions with SVE ACLE parameters. Clang will use the old
  manglings if ``-fclang-abi-compat=17`` or lower is  specified.

- New AArch64 asm constraints have been added for r8-r11(Uci) and r12-r15(Ucj).

- Support has been added for the following processors (-mcpu identifiers in parenthesis):

  For Arm:

  * Cortex-M52 (cortex-m52).

  For AArch64:

  * Cortex-A520 (cortex-a520).
  * Cortex-A720 (cortex-a720).
  * Cortex-X4 (cortex-x4).

Android Support
^^^^^^^^^^^^^^^

- Android target triples are usually suffixed with a version. Clang searches for
  target-specific runtime and standard libraries in directories named after the
  target (e.g. if you're building with ``--target=aarch64-none-linux-android21``,
  Clang will look for ``lib/aarch64-none-linux-android21`` under its resource
  directory to find runtime libraries). If an exact match isn't found, Clang
  would previously fall back to a directory without any version (which would be
  ``lib/aarch64-none-linux-android`` in our example). Clang will now look for
  directories for lower versions and use the newest version it finds instead,
  e.g. if you have ``lib/aarch64-none-linux-android21`` and
  ``lib/aarch64-none-linux-android29``, ``-target aarch64-none-linux-android23``
  will use the former and ``-target aarch64-none-linux-android30`` will use the
  latter. Falling back to a versionless directory will now emit a warning, and
  the fallback will be removed in Clang 19.

Windows Support
^^^^^^^^^^^^^^^
- Fixed an assertion failure that occurred due to a failure to propagate
  ``MSInheritanceAttr`` attributes to class template instantiations created
  for explicit template instantiation declarations.

- The ``-fno-auto-import`` option was added for MinGW targets. The option both
  affects code generation (inhibiting generating indirection via ``.refptr``
  stubs for potentially auto imported symbols, generating smaller and more
  efficient code) and linking (making the linker error out on such cases).
  If the option only is used during code generation but not when linking,
  linking may succeed but the resulting executables may expose issues at
  runtime.

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^
- Unaligned memory accesses can be toggled by ``-m[no-]unaligned-access`` or the
  aliases ``-m[no-]strict-align``.
- CodeGen of RV32E/RV64E was supported experimentally.
- CodeGen of ilp32e/lp64e was supported experimentally.

- Default ABI with F but without D was changed to ilp32f for RV32 and to lp64f
  for RV64.

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA Support
^^^^^^^^^^^^

- Clang now supports CUDA SDK up to 12.3
- Added support for sm_90a

AIX Support
^^^^^^^^^^^

- Introduced the ``-maix-small-local-exec-tls`` option to produce a faster
  access sequence for local-exec TLS variables where the offset from the TLS
  base is encoded as an immediate operand.
  This access sequence is not used for TLS variables larger than 32KB, and is
  currently only supported on 64-bit mode.

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
- ``__builtin_isfpclass`` builtin now supports vector types.
- ``#pragma float_control(precise,on)`` enables precise floating-point
  semantics. If ``math-errno`` is disabled in the current TU, clang will
  re-enable ``math-errno`` in the presense of
  ``#pragma float_control(precise,on)``.
- Add ``__builtin_exp10``, ``__builtin_exp10f``,
  ``__builtin_exp10f16``, ``__builtin_exp10l`` and
  ``__builtin_exp10f128`` builtins.
- Add ``__builtin_iszero``, ``__builtin_issignaling`` and
  ``__builtin_issubnormal``.
- Add support for C99's ``#pragma STDC CX_LIMITED_RANGE`` feature.  This
  enables the naive mathematical formulas for complex multiplication and
  division, which are faster but do not correctly handle overflow and infinities.

AST Matchers
------------
- Add ``convertVectorExpr``.
- Add ``dependentSizedExtVectorType``.
- Add ``macroQualifiedType``.
- Add ``CXXFoldExpr`` related matchers: ``cxxFoldExpr``, ``callee``,
  ``hasInit``, ``hasPattern``, ``isRightFold``, ``isLeftFold``,
  ``isUnaryFold``, ``isBinaryFold``, ``hasOperator``, ``hasLHS``, ``hasRHS``, ``hasEitherOperand``.

clang-format
------------
- Add ``AllowBreakBeforeNoexceptSpecifier`` option.
- Add ``AllowShortCompoundRequirementOnASingleLine`` option.
- Change ``BreakAfterAttributes`` from ``Never`` to ``Leave`` in LLVM style.
- Add ``BreakAdjacentStringLiterals`` option.
- Add ``ObjCPropertyAttributeOrder`` which can be used to sort ObjC property
  attributes (like ``nonatomic, strong, nullable``).
- Add ``PenaltyBreakScopeResolution`` option.
- Add ``.clang-format-ignore`` files.
- Add ``AlignFunctionPointers`` sub-option for ``AlignConsecutiveDeclarations``.

libclang
--------

- Exposed arguments of ``clang::annotate``.
- ``clang::getCursorKindForDecl`` now recognizes linkage specifications such as
  ``extern "C"`` and reports them as ``CXCursor_LinkageSpec``.

Static Analyzer
---------------

New features
^^^^^^^^^^^^

- Implemented the ``[[clang::suppress]]`` attribute for suppressing diagnostics
  of static analysis tools, such as the Clang Static Analyzer.
  `Documentation <https://clang.llvm.org/docs/AttributeReference.html#suppress>`__.

- Added support for the ``cleanup`` attribute.
  `Documentation <https://clang.llvm.org/docs/AttributeReference.html#cleanup>`__.

- Support "Deducing this" (P0847R7). (Worked out of the box)
  (`af4751738db8 <https://github.com/llvm/llvm-project/commit/af4751738db89a142a8880c782d12d4201b222a8>`__)

- Added a new checker ``core.BitwiseShift`` which reports situations where
  bitwise shift operators produce undefined behavior (because some operand is
  negative or too large).
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#core-bitwiseshift-c-c>`__.

- Added a new experimental checker ``alpha.core.StdVariant`` to detect variant
  accesses via wrong alternatives.
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#alpha-core-stdvariant-c>`__.
  (`#66481 <https://github.com/llvm/llvm-project/pull/66481>`_)

- Added a new experimental checker ``alpha.cplusplus.ArrayDelete`` to detect
  destructions of arrays of polymorphic objects that are destructed as their
  base class (`CERT EXP51-CPP <https://wiki.sei.cmu.edu/confluence/display/cplusplus/EXP51-CPP.+Do+not+delete+an+array+through+a+pointer+of+the+incorrect+type>`_).
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#alpha-cplusplus-arraydelete-c>`__.
  (`0e246bb67573 <https://github.com/llvm/llvm-project/commit/0e246bb67573799409d0085b89902a330998ddcc>`_)

- Added a new checker configuration option ``InvalidatingGetEnv=[true,false]`` to
  ``security.cert.env.InvalidPtr``. It's not set by default.
  If set, ``getenv`` calls won't invalidate previously returned pointers.
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#security-cert-env-invalidptr>`__.
  (`#67663 <https://github.com/llvm/llvm-project/pull/67663>`_)

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^

- Fixed a crash caused by ``builtin_bit_cast``.
  (`#69922 <https://github.com/llvm/llvm-project/issues/69922>`_)

- Fixed a ``core.StackAddressEscape`` crash on temporary object fields.
  (`#66221 <https://github.com/llvm/llvm-project/issues/66221>`_)

- A few crashes have been found and fixed using randomized testing related
  to the use of ``_BitInt()`` in tidy checks and in clang analysis.
  (`#67212 <https://github.com/llvm/llvm-project/pull/67212>`_,
  `#66782 <https://github.com/llvm/llvm-project/pull/66782>`_,
  `#65889 <https://github.com/llvm/llvm-project/pull/65889>`_,
  `#65888 <https://github.com/llvm/llvm-project/pull/65888>`_,
  `#65887 <https://github.com/llvm/llvm-project/pull/65887>`_)

- Fixed note links of the HTML output.
  (`#64054 <https://github.com/llvm/llvm-project/issues/64054>`_)

- Allow widening rage-based for loops.
  (`#70190 <https://github.com/llvm/llvm-project/pull/70190>`_)

- Fixed uninitialized base class with initializer list when ctor is not
  declared in the base class.
  (`#70464 <https://github.com/llvm/llvm-project/issues/70464>`_,
  `#59493 <https://github.com/llvm/llvm-project/issues/59493>`_,
  `#54533 <https://github.com/llvm/llvm-project/issues/54533>`_)

- Fixed an ``alpha.unix.cstring`` crash on variadic functions.
  (`#74269 <https://github.com/llvm/llvm-project/issues/74269>`_)

- Fix false positive in mutation check when using pointer to member function.
  (`#66204 <https://github.com/llvm/llvm-project/issues/66204>`_)

Improvements
^^^^^^^^^^^^

- Improved the ``unix.StdCLibraryFunctions`` checker by modeling more
  functions like ``send``, ``recv``, ``readlink``, ``fflush``, ``mkdtemp``,
  ``getcwd`` and ``errno`` behavior.
  (`52ac71f92d38 <https://github.com/llvm/llvm-project/commit/52ac71f92d38f75df5cb88e9c090ac5fd5a71548>`_,
  `#77040 <https://github.com/llvm/llvm-project/pull/77040>`_,
  `#76671 <https://github.com/llvm/llvm-project/pull/76671>`_,
  `#71373 <https://github.com/llvm/llvm-project/pull/71373>`_,
  `#76557 <https://github.com/llvm/llvm-project/pull/76557>`_,
  `#71392 <https://github.com/llvm/llvm-project/pull/71392>`_)

- Fixed a false negative for when accessing a nonnull property (ObjC).
  (`1dceba3a3684 <https://github.com/llvm/llvm-project/commit/1dceba3a3684d12394731e09a6cf3efcebf07a3a>`_)

- ``security.insecureAPI.DeprecatedOrUnsafeBufferHandling`` now considers
  ``fprintf`` calls unsafe.
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-deprecatedorunsafebufferhandling-c>`__.

- Improved the diagnostics of the ``optin.core.EnumCastOutOfRange`` checker.
  It will display the name and the declaration of the enumeration along with
  the concrete value being cast to the enum.
  (`#74503 <https://github.com/llvm/llvm-project/pull/74503>`_)

- Improved the ``alpha.security.ArrayBoundV2`` checker for detecting buffer
  accesses prior the buffer; and also reworked the diagnostic messages.
  (`3e014038b373 <https://github.com/llvm/llvm-project/commit/3e014038b373e5a4a96d89d46cea17e4d2456a04>`_,
  `#70056 <https://github.com/llvm/llvm-project/pull/70056>`_,
  `#72107 <https://github.com/llvm/llvm-project/pull/72107>`_)

- Improved the ``alpha.unix.cstring.OutOfBounds`` checking both ends of the
  buffers in more cases.
  (`c3a87ddad62a <https://github.com/llvm/llvm-project/commit/c3a87ddad62a6cc01acaccc76592bc6730c8ac3c>`_,
  `0954dc3fb921 <https://github.com/llvm/llvm-project/commit/0954dc3fb9214b994623f5306473de075f8e3593>`_)

- Improved the ``alpha.unix.Stream`` checker by modeling more functions
  ``fputs``, ``fputc``, ``fgets``, ``fgetc``, ``fdopen``, ``ungetc``, ``fflush``
  and no not recognize alternative ``fopen`` and ``tmpfile`` implementations.
  (`#76776 <https://github.com/llvm/llvm-project/pull/76776>`_,
  `#74296 <https://github.com/llvm/llvm-project/pull/74296>`_,
  `#73335 <https://github.com/llvm/llvm-project/pull/73335>`_,
  `#72627 <https://github.com/llvm/llvm-project/pull/72627>`_,
  `#71518 <https://github.com/llvm/llvm-project/pull/71518>`_,
  `#72016 <https://github.com/llvm/llvm-project/pull/72016>`_,
  `#70540 <https://github.com/llvm/llvm-project/pull/70540>`_,
  `#73638 <https://github.com/llvm/llvm-project/pull/73638>`_,
  `#77331 <https://github.com/llvm/llvm-project/pull/77331>`_)

- The ``alpha.security.taint.TaintPropagation`` checker no longer propagates
  taint on ``strlen`` and ``strnlen`` calls, unless these are marked
  explicitly propagators in the user-provided taint configuration file.
  This removal empirically reduces the number of false positive reports.
  Read the PR for the details.
  (`#66086 <https://github.com/llvm/llvm-project/pull/66086>`_)

- Other taint-related improvements.
  (`#66358 <https://github.com/llvm/llvm-project/pull/66358>`_,
  `#66074 <https://github.com/llvm/llvm-project/pull/66074>`_,
  `#66358 <https://github.com/llvm/llvm-project/pull/66358>`_)

- Checkers can query constraint bounds to improve diagnostic messages.
  (`#74141 <https://github.com/llvm/llvm-project/pull/74141>`_)

Moved checkers
^^^^^^^^^^^^^^

- Move checker ``alpha.unix.Errno`` out of the ``alpha`` package
  to ``unix.Errno``.
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#unix-errno-c>`__.

- Move checker ``alpha.unix.StdCLibraryFunctions`` out of the ``alpha`` package
  to ``unix.StdCLibraryFunctions``.
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#unix-stdclibraryfunctions-c>`__.

- Move checker ``alpha.security.cert.env.InvalidPtr`` out of the ``alpha``
  package to ``security.cert.env.InvalidPtr``.
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#security-cert-env-invalidptr>`__.

- Move checker ``alpha.cplusplus.EnumCastOutOfRange`` out of the ``alpha``
  package to ``optin.core.EnumCastOutOfRange``.
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#optin-core-enumcastoutofrange-c-c>`__.

.. _release-notes-sanitizers:

Sanitizers
----------

- ``-fsanitize=signed-integer-overflow`` now instruments ``__builtin_abs`` and
  ``abs`` builtins.

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
