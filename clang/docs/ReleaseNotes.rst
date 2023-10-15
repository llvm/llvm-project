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

- The default extension name for PCH generation (``-c -xc-header`` and ``-c
  -xc++-header``) is now ``.pch`` instead of ``.gch``.
- ``-include a.h`` probing ``a.h.gch`` is deprecated. Change the extension name
  to ``.pch`` or use ``-include-pch a.h.gch``.
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

ABI Changes in This Version
---------------------------
- Following the SystemV ABI for x86-64, ``__int128`` arguments will no longer
  be split between a register and a stack slot.

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


Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C Language Changes
------------------
- ``structs``, ``unions``, and ``arrays`` that are const may now be used as
  constant expressions.  This change is more consistent with the behavior of
  GCC.
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

Non-comprehensive list of changes in this release
-------------------------------------------------

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

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

* ``-Woverriding-t-option`` is renamed to ``-Woverriding-option``.
* ``-Winterrupt-service-routine`` is renamed to ``-Wexcessive-regsave`` as a generalization
* ``-frewrite-includes`` now guards the original #include directives with
  ``__CLANG_REWRITTEN_INCLUDES``, and ``__CLANG_REWRITTEN_SYSTEM_INCLUDES`` as
  appropriate.

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

Improvements to Clang's diagnostics
-----------------------------------
- Clang constexpr evaluator now prints template arguments when displaying
  template-specialization function calls.
- Clang contexpr evaluator now displays notes as well as an error when a constructor
  of a base class is not called in the constructor of its derived class.
- Clang no longer emits ``-Wmissing-variable-declarations`` for variables declared
  with the ``register`` storage class.
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
- The fix-it emitted by ``-Wformat`` for scoped enumerations now take the
  enumeration's underlying type into account instead of suggesting a type just
  based on the format string specifier being used.
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
- Fixed an issue when a shift count larger than ``__INT64_MAX__``, in a right
  shift operation, could result in missing warnings about
  ``shift count >= width of type`` or internal compiler error.
- Fixed an issue with computing the common type for the LHS and RHS of a `?:`
  operator in C. No longer issuing a confusing diagnostic along the lines of
  "incompatible operand types ('foo' and 'foo')" with extensions such as matrix
  types. Fixes (`#69008 <https://github.com/llvm/llvm-project/issues/69008>`_)
- Fix a crash when evaluating comparasion between the field from the same variable
  which initialized from compound literals in C. Fixes
  (`#69065 <https://github.com/llvm/llvm-project/issues/69065>`_)

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
  (`#64138 <https://github.com/llvm/llvm-project/issues/64138>`_).

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

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed an import failure of recursive friend class template.
  `Issue 64169 <https://github.com/llvm/llvm-project/issues/64169>`_
- Remove unnecessary RecordLayout computation when importing UnaryOperator. The
  computed RecordLayout is incorrect if fields are not completely imported and
  should not be cached.
  `Issue 64170 <https://github.com/llvm/llvm-project/issues/64170>`_

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

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

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

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA Support
^^^^^^^^^^^^

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

AST Matchers
------------
- Add ``convertVectorExpr``.
- Add ``dependentSizedExtVectorType``.
- Add ``macroQualifiedType``.

clang-format
------------
- Add ``AllowBreakBeforeNoexceptSpecifier`` option.

libclang
--------

- Exposed arguments of ``clang::annotate``.

Static Analyzer
---------------

- Added a new checker ``core.BitwiseShift`` which reports situations where
  bitwise shift operators produce undefined behavior (because some operand is
  negative or too large).

- Fix false positive in mutation check when using pointer to member function.
  (`#66204: <https://github.com/llvm/llvm-project/issues/66204>`_).

- The ``alpha.security.taint.TaintPropagation`` checker no longer propagates
  taint on ``strlen`` and ``strnlen`` calls, unless these are marked
  explicitly propagators in the user-provided taint configuration file.
  This removal empirically reduces the number of false positive reports.
  Read the PR for the details.
  (`#66086 <https://github.com/llvm/llvm-project/pull/66086>`_)

- A few crashes have been found and fixed using randomized testing related
  to the use of ``_BitInt()`` in tidy checks and in clang analysis. See
  `#67212 <https://github.com/llvm/llvm-project/pull/67212>`_,
  `#66782 <https://github.com/llvm/llvm-project/pull/66782>`_,
  `#65889 <https://github.com/llvm/llvm-project/pull/65889>`_,
  `#65888 <https://github.com/llvm/llvm-project/pull/65888>`_, and
  `#65887 <https://github.com/llvm/llvm-project/pull/65887>`_

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
