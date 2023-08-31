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

- ``__builtin_object_size`` and ``__builtin_dynamic_object_size`` now add the
  ``sizeof`` the elements specified in designated initializers of flexible
  array members for structs that contain them. This change is more consistent
  with the behavior of GCC.

C++ Specific Potentially Breaking Changes
-----------------------------------------
- Clang won't search for coroutine_traits in std::experimental namespace any more.
  Clang will only search for std::coroutine_traits for coroutines then.
- Clang no longer allows dereferencing of a ``void *`` as an extension. Clang 16
  converted this to a default-error as ``-Wvoid-ptr-dereference``, as well as a
  SFINAE error. This flag is still valid however, as it disables the equivalent
  warning in C.

ABI Changes in This Version
---------------------------
- A bug in evaluating the ineligibility of some special member functions has been fixed. This can
  make some classes trivially copyable that were not trivially copyable before. (`#62555 <https://github.com/llvm/llvm-project/issues/62555>`_)

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
- Declaring namespace std to be an inline namespace is now prohibited, `[namespace.std]p7`.
- Improved code generation for ``dynamic_cast`` to a ``final`` type. Instead of
  dispatching to the runtime library to compare the RTTI data, Clang now
  generates a direct comparison of the vtable pointer in cases where the ABI
  requires the vtable for a class to be unique. This optimization can be
  disabled with ``-fno-assume-unique-vtables``. This optimization is not yet
  implemented for the MS C++ ABI.

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Implemented the rule introduced by `CA104 <https://wg21.link/P2103R0>`_  for comparison of
  constraint-expressions. Improved support for out-of-line definitions of constrained templates.
  This fixes:
  `#49620 <https://github.com/llvm/llvm-project/issues/49620>`_,
  `#60231 <https://github.com/llvm/llvm-project/issues/60231>`_,
  `#61414 <https://github.com/llvm/llvm-project/issues/61414>`_,
  `#61809 <https://github.com/llvm/llvm-project/issues/61809>`_.
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
- Clang now implements `[temp.deduct]p9`. Substitution failures inside lambdas from
  unevaluated contexts will be surfaced as errors. They were previously handled as
  SFINAE.
- Clang now supports `requires cplusplus20` for module maps.
- Implemented missing parts of `P2002R1: Consistent comparison operators <https://wg21.link/P2002R1>`_
- Clang now defines `__cpp_consteval` macro.
- Implemented `P1816R0: <https://wg21.link/p1816r0>`_ and `P2082R1: <https://wg21.link/p2082r1>`_,
  which allows CTAD for aggregates.

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
  functions. Which include allowing non-literal types as return values and parameters, allow calling of
  non-constexpr functions and constructors.
- Clang now supports `requires cplusplus23` for module maps.
- Implemented `P2564R3: consteval needs to propagate up <https://wg21.link/P2564R3>`_.

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Compiler flags ``-std=c++2c`` and ``-std=gnu++2c`` have been added for experimental C++2c implementation work.
- Implemented `P2738R1: constexpr cast from void* <https://wg21.link/P2738R1>`_.
- Partially implemented `P2361R6: Unevaluated strings <https://wg21.link/P2361R6>`_.
  The changes to attributes declarations are not part of this release.
- Implemented `P2741R3: user-generated static_assert messages  <https://wg21.link/P2741R3>`_.

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Implemented `DR2397 <https://wg21.link/CWG2397>`_ which allows ``auto`` specifier for pointers
  and reference to arrays.
- Implemented `CWG2521 <https://wg21.link/CWG2521>`_ which reserves using ``__`` in user-defined
  literal suffixes and deprecates literal operator function declarations using an identifier.
  Taught ``-Wuser-defined-literals`` for the former, on by default, and added
  ``-Wdeprecated-literal-operator`` for the latter, off by default for now.

  .. code-block:: c++

    // What follows is warned by -Wuser-defined-literals
    // albeit "ill-formed, no diagnostic required".
    // Its behavior is undefined, [reserved.names.general]p2.
    string operator ""__i18n(const char*, std::size_t);

    // Assume this declaration is not in the global namespace.
    // -Wdeprecated-literal-operator diagnoses the extra space.
    string operator "" _i18n(const char*, std::size_t);
    //                ^ an extra space

C Language Changes
------------------
- Support for outputs from asm goto statements along indirect edges has been
  added. (`#53562 <https://github.com/llvm/llvm-project/issues/53562>`_)
- Fixed a bug that prevented initialization of an ``_Atomic``-qualified pointer
  from a null pointer constant.
- Fixed a bug that prevented casting to an ``_Atomic``-qualified type.
  (`#39596 <https://github.com/llvm/llvm-project/issues/39596>`_)
- Added an extension to ``_Generic`` which allows the first operand to be a
  type rather than an expression. The type does not undergo any conversions,
  which makes this feature suitable for matching qualified types, incomplete
  types, and function or array types.

  .. code-block:: c

    const int i = 12;
    _Generic(i, int : 0, const int : 1); // Warns about unreachable code, the
                                         // result is 0, not 1.
    _Generic(typeof(i), int : 0, const int : 1); // Result is 1, not 0.
- ``structs``, ``unions``, and ``arrays`` that are const may now be used as
  constant expressions.  This change is more consistent with the behavior of
  GCC.

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

- Implemented `WG14 N3124 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3124.pdf>_`,
  which allows any universal character name to appear in character and string literals.


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
- A new builtin type trait ``__is_trivially_equality_comparable`` has been added,
  which checks whether comparing two instances of a type is equivalent to
  ``memcmp(&lhs, &rhs, sizeof(T)) == 0``.
- Clang now ignores null directives outside of the include guard when deciding
  whether a file can be enabled for the multiple-include optimization.
- Clang now support ``__builtin_FUNCSIG()`` which returns the same information
  as the ``__FUNCSIG__`` macro (available only with ``-fms-extensions`` flag).
  This fixes (`#58951 <https://github.com/llvm/llvm-project/issues/58951>`_).
- Clang now supports the `NO_COLOR <https://no-color.org/>`_ environment
  variable as a way to disable color diagnostics.
- Clang now supports ``__builtin_isfpclass``, which checks if the specified
  floating-point value falls into any of the specified data classes.
- Added ``__builtin_elementwise_round`` for  builtin for floating
  point types. This allows access to ``llvm.round`` for
  arbitrary floating-point and vector of floating-point types.
- Added ``__builtin_elementwise_rint`` for floating point types. This
  allows access to ``llvm.rint`` for arbitrary floating-point and
  vector of floating-point types.
- Added ``__builtin_elementwise_nearbyint`` for floating point
  types. This allows access to ``llvm.nearbyint`` for arbitrary
  floating-point and vector of floating-point types.
- Clang AST matcher now matches concept declarations with `conceptDecl`.

New Compiler Flags
------------------
- The flag ``-std=c++23`` has been added. This behaves the same as the existing
  flag ``-std=c++2b``.
- ``-dumpdir`` has been implemented to specify auxiliary and dump output
  filenames for features like ``-gsplit-dwarf``.
- ``-fcaret-diagnostics-max-lines=`` has been added as a driver options, which
  lets users control the maximum number of source lines printed for a
  caret diagnostic.
- ``-fkeep-persistent-storage-variables`` has been implemented to keep all
  variables that have a persistent storage duration—including global, static
  and thread-local variables—to guarantee that they can be directly addressed.
  Since this inhibits the merging of the affected variables, the number of
  individual relocations in the program will generally increase.
- ``-f[no-]assume-unique-vtables`` controls whether Clang assumes that each
  class has a unique vtable address, when that is required by the ABI.
- ``-print-multi-flags-experimental`` prints the flags used for multilib
  selection. See `the multilib docs <https://clang.llvm.org/docs/Multilib.html>`_
  for more details.


Deprecated Compiler Flags
-------------------------

- ``-fdouble-square-bracket-attributes`` has been deprecated. It is ignored now
  and will be removed in Clang 18.

Modified Compiler Flags
-----------------------

- ``clang -g -gsplit-dwarf a.c -o obj/x`` (compile and link) now generates the
  ``.dwo`` file at ``obj/x-a.dwo``, instead of a file in the temporary
  directory (``/tmp`` on \*NIX systems, if none of the environment variables
  TMPDIR, TMP, and TEMP are specified).

Removed Compiler Flags
-------------------------
- The deprecated flag `-fmodules-ts` is removed. Please use ``-std=c++20``
  or higher to use standard C++ modules instead.
- The deprecated flag `-fcoroutines-ts` is removed. Please use ``-std=c++20``
  or higher to use standard C++ coroutines instead.
- The CodeGen flag `-lower-global-dtors-via-cxa-atexit` which affects how global
  destructors are lowered for MachO is removed without replacement. The default
  of `-lower-global-dtors-via-cxa-atexit=true` is now the only supported way.
- The cc1 flag ``-no-opaque-pointers`` has been removed.

Attribute Changes in Clang
--------------------------
- Introduced a new function attribute ``__attribute__((unsafe_buffer_usage))``
  to be worn by functions containing buffer operations that could cause out of
  bounds memory accesses. It emits warnings at call sites to such functions when
  the flag ``-Wunsafe-buffer-usage`` is enabled.
- ``__declspec`` attributes can now be used together with the using keyword. Before
  the attributes on ``__declspec`` was ignored, while now it will be forwarded to the
  point where the alias is used. Note, some incorrect uses of ``__declspec`` on a
  ``using`` declaration were being silently ignored and will now be appropriately
  diagnosed as ignoring the attribute.
- Introduced a new ``USR`` (unified symbol resolution) clause inside of the
  existing ``__attribute__((external_source_symbol))`` attribute. Clang's indexer
  uses the optional USR value when indexing Clang's AST. This value is expected
  to be generated by an external compiler when generating C++ bindings during
  the compilation of the foreign language sources (e.g. Swift).
- The ``__has_attribute``, ``__has_c_attribute`` and ``__has_cpp_attribute``
  preprocessor operators now return 1 also for attributes defined by plugins.
- Improve the AST fidelity of ``alignas`` and ``_Alignas`` attribute. Before, we
  model ``alignas(type-id)`` as though the user wrote ``alignas(alignof(type-id))``,
  now we directly use ``alignas(type-id)``.

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
- Clang now correctly diagnoses statement attributes ``[[clang::always_inline]]`` and
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
- Address a false positive in ``-Wpacked`` when applied to a non-pod type using
  Clang ABI >= 15.
  (`#62353: <https://github.com/llvm/llvm-project/issues/62353>`_,
  fallout from the non-POD packing ABI fix in LLVM 15).
- Clang constexpr evaluator now prints subobject's name instead of its type in notes
  when a constexpr variable has uninitialized subobjects after its constructor call.
  (`#58601 <https://github.com/llvm/llvm-project/issues/58601>`_)
- Clang's `-Wshadow` warning now warns about shadowings by static local variables
  (`#62850: <https://github.com/llvm/llvm-project/issues/62850>`_).
- Clang now warns when any predefined macro is undefined or redefined, instead
  of only some of them.
- Clang now correctly diagnoses when the argument to ``alignas`` or ``_Alignas``
  is an incomplete type.
  (`#55175: <https://github.com/llvm/llvm-project/issues/55175>`_, and fixes an
  incorrect mention of ``alignof`` in a diagnostic about ``alignas``).
- Clang will now show a margin with line numbers to the left of each line
  of code it prints for diagnostics. This can be disabled using
  ``-fno-diagnostics-show-line-numbers``. At the same time, the maximum
  number of code lines it prints has been increased from 1 to 16. This
  can be controlled using ``-fcaret-diagnostics-max-lines=``.
- Clang no longer emits ``-Wunused-variable`` warnings for variables declared
  with ``__attribute__((cleanup(...)))`` to match GCC's behavior.
- Clang now issues expected warnings for situations of comparing with NULL pointers.
  (`#42992: <https://github.com/llvm/llvm-project/issues/42992>`_)
- Clang now diagnoses unused const-qualified variable template as
  "unused variable template" rather than "unused variable".
- When diagnosing a constant expression where an enum without a fixed underlying
  type is set to a value outside the range of the enum's values, clang will now
  print the name of the enum in question.
- Clang no longer diagnoses a read of an empty structure as use of an
  uninitialized variable.
  (`#26842: <https://github.com/llvm/llvm-project/issues/26842>`_)
- The Fix-It emitted for unused labels used to expand to the next line, which caused
  visual oddities now that Clang shows more than one line of code snippet. This has
  been fixed and the Fix-It now only spans to the end of the ``:``.
- Clang now underlines the parameter list of function declaration when emitting
  a note about the mismatch in the number of arguments.
- Clang now diagnoses unexpected tokens after a
  ``#pragma clang|GCC diagnostic push|pop`` directive.
  (`#13920: <https://github.com/llvm/llvm-project/issues/13920>`_)
- Clang now does not try to analyze cast validity on variables with dependent alignment (`#63007: <https://github.com/llvm/llvm-project/issues/63007>`_).
- Clang constexpr evaluator now displays member function calls more precisely
  by making use of the syntactical structure of function calls. This avoids display
  of syntactically invalid codes in diagnostics.
  (`#57081: <https://github.com/llvm/llvm-project/issues/57081>`_)
- Clang no longer emits inappropriate notes about the loss of ``__unaligned`` qualifier
  on overload resolution, when the actual reason for the failure is loss of other qualifiers.
- The note emitted when an ``operator==`` was defaulted as deleted used to refer to
  the lack of a data member's "three-way comparison operator". It now refers correctly
  to the data member's ``operator==``.
  (`#63960: <https://github.com/llvm/llvm-project/issues/63960>`_)
- Clang's notes about unconvertible types in overload resolution failure now covers
  the source range of parameter declaration of the candidate function declaration.

  *Example Code*:

  .. code-block:: c++

     void func(int aa, int bb);
     void test() { func(1, "two"); }

  *BEFORE*:

  .. code-block:: text

    source:2:15: error: no matching function for call to 'func'
    void test() { func(1, "two");  }
                  ^~~~
    source:1:6: note: candidate function not viable: no known conversion from 'const char[4]' to 'int' for 2nd argument
    void func(int aa, int bb);
         ^

  *AFTER*:

  .. code-block:: text

    source:2:15: error: no matching function for call to 'func'
    void test() { func(1, "two");  }
                  ^~~~
    source:1:6: note: candidate function not viable: no known conversion from 'const char[4]' to 'int' for 2nd argument
    void func(int aa, int bb);
         ^            ~~~~~~

- ``-Wformat`` cast fix-its will now suggest ``static_cast`` instead of C-style casts
  for C++ code.
- ``-Wformat`` will no longer suggest a no-op fix-it for fixing scoped enum format
  warnings. Instead, it will suggest casting the enum object to the type specified
  in the format string.
- Clang now emits ``-Wconstant-logical-operand`` warning even when constant logical
  operand is on left side.
  (`#37919 <https://github.com/llvm/llvm-project/issues/37919>`_)
- Clang contexpr evaluator now displays notes as well as an error when a constructor
  of a base class is not called in the constructor of its derived class.

Bug Fixes in This Version
-------------------------
- Fixed an issue where a class template specialization whose declaration is
  instantiated in one module and whose definition is instantiated in another
  module may end up with members associated with the wrong declaration of the
  class, which can result in miscompiles in some cases.
- Added a new diagnostic warning group
  ``-Wdeprecated-redundant-constexpr-static-def``, under the existing
  ``-Wdeprecated`` group. This controls warnings about out-of-line definitions
  of 'static constexpr' data members that are unnecessary from C++17 onwards.
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
- Fix coroutines issue where ``get_return_object()`` result was always eagerly
  converted to the return type. Eager initialization (allowing RVO) is now only
  performed when these types match, otherwise deferred initialization is used,
  enabling short-circuiting coroutines use cases. This fixes
  (`#56532 <https://github.com/llvm/llvm-project/issues/56532>`_) in
  anticipation of `CWG2563 <https://cplusplus.github.io/CWG/issues/2563.html>_`.
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
- Fix crash when handling initialization candidates for invalid deduction guide.
  (`#62408 <https://github.com/llvm/llvm-project/issues/62408>`_)
- Fix crash when redefining a variable with an invalid type again with an
  invalid type. (`#62447 <https://github.com/llvm/llvm-project/issues/62447>`_)
- Fix a stack overflow issue when evaluating ``consteval`` default arguments.
  (`#60082 <https://github.com/llvm/llvm-project/issues/60082>`_)
- Fix the assertion hit when generating code for global variable initializer of
  _BitInt(1) type.
  (`#62207 <https://github.com/llvm/llvm-project/issues/62207>`_)
- Fix lambdas and other anonymous function names not respecting ``-fdebug-prefix-map``
  (`#62192 <https://github.com/llvm/llvm-project/issues/62192>`_)
- Fix crash when attempting to pass a non-pointer type as first argument of
  ``__builtin_assume_aligned``.
  (`#62305 <https://github.com/llvm/llvm-project/issues/62305>`_)
- A default argument for a non-type template parameter is evaluated and checked
  at the point where it is required. This fixes:
  (`#62224 <https://github.com/llvm/llvm-project/issues/62224>`_) and
  (`#62596 <https://github.com/llvm/llvm-project/issues/62596>`_)
- Fix an assertion when instantiating the body of a Class Template Specialization
  when it had been instantiated from a partial template specialization with different
  template arguments on the containing class. This fixes:
  (`#60778 <https://github.com/llvm/llvm-project/issues/60778>`_).
- Fix a crash when an enum constant has a dependent-type recovery expression for
  C.
  (`#62446 <https://github.com/llvm/llvm-project/issues/62446>`_).
- Propagate the value-dependent bit for VAArgExpr. Fixes a crash where a
  __builtin_va_arg call has invalid arguments.
  (`#62711 <https://github.com/llvm/llvm-project/issues/62711>`_).
- Fix crash on attempt to initialize union with flexible array member.
  (`#61746 <https://github.com/llvm/llvm-project/issues/61746>`_).
- Clang `TextNodeDumper` enabled through `-ast-dump` flag no longer evaluates the
  initializer of constexpr `VarDecl` if the declaration has a dependent type.
- Match GCC's behavior for ``__builtin_object_size`` and
  ``__builtin_dynamic_object_size`` on structs containing flexible array
  members.
  (`#62789 <https://github.com/llvm/llvm-project/issues/62789>`_).
- Fix a crash when instantiating a non-type template argument in a dependent scope.
  (`#62533 <https://github.com/llvm/llvm-project/issues/62533>`_).
- Fix crash when diagnosing default comparison method.
  (`#62791 <https://github.com/llvm/llvm-project/issues/62791>`_) and
  (`#62102 <https://github.com/llvm/llvm-project/issues/62102>`_).
- Fix crash when passing a braced initializer list to a parentehsized aggregate
  initialization expression.
  (`#63008 <https://github.com/llvm/llvm-project/issues/63008>`_).
- Reject increment of bool value in unevaluated contexts after C++17.
  (`#47517 <https://github.com/llvm/llvm-project/issues/47517>`_).
- Fix assertion and quality of diagnostic messages in a for loop
  containing multiple declarations and a range specifier
  (`#63010 <https://github.com/llvm/llvm-project/issues/63010>`_).
- Fix rejects-valid when consteval operator appears inside of a template.
  (`#62886 <https://github.com/llvm/llvm-project/issues/62886>`_).
- Fix crash for code using ``_Atomic`` types in C++
  (`See patch <https://reviews.llvm.org/D152303>`_).
- Fix crash when passing a value larger then 64 bits to the aligned attribute.
  (`#50534 <https://github.com/llvm/llvm-project/issues/50534>`_).
- CallExpr built for C error-recovery now is always type-dependent. Fixes a
  crash when we encounter a unresolved TypoExpr during diagnostic emission.
  (`#50244 <https://github.com/llvm/llvm-project/issues/50244>`_).
- Apply ``-fmacro-prefix-map`` to anonymous tags in template arguments
  (`#63219 <https://github.com/llvm/llvm-project/issues/63219>`_).
- Clang now properly diagnoses format string mismatches involving scoped
  enumeration types. A scoped enumeration type is not promoted to an integer
  type by the default argument promotions, and thus this is UB. Clang's
  behavior now matches GCC's behavior in C++.
  (`#38717 <https://github.com/llvm/llvm-project/issues/38717>`_).
- Fixed a failing assertion when implicitly defining a function within a GNU
  statement expression that appears outside of a function block scope. The
  assertion was benign outside of asserts builds and would only fire in C.
  (`#48579 <https://github.com/llvm/llvm-project/issues/48579>`_).
- Fixed a failing assertion when applying an attribute to an anonymous union.
  The assertion was benign outside of asserts builds and would only fire in C++.
  (`#48512 <https://github.com/llvm/llvm-project/issues/48512>`_).
- Fixed a failing assertion when parsing incomplete destructor.
  (`#63503 <https://github.com/llvm/llvm-project/issues/63503>`_)
- Fix C++17 mode assert when parsing malformed code and the compiler is
  attempting to see if it could be type template for class template argument
  deduction. This fixes
  (`Issue 57495 <https://github.com/llvm/llvm-project/issues/57495>`_)
- Fix missing destructor calls and therefore memory leaks in generated code
  when an immediate invocation appears as a part of an expression that produces
  temporaries.
  (`#60709 <https://github.com/llvm/llvm-project/issues/60709>`_).
- Fixed a missed integer overflow warning with temporary values.
  (`#63629 <https://github.com/llvm/llvm-project/issues/63629>`_)
- Fixed parsing of elaborated type specifier inside of a new expression.
  (`#34341 <https://github.com/llvm/llvm-project/issues/34341>`_)
- Clang now correctly evaluates ``__has_extension (cxx_defaulted_functions)``
  and ``__has_extension (cxx_default_function_template_args)`` to 1.
  (`#61758 <https://github.com/llvm/llvm-project/issues/61758>`_)
- Stop evaluating a constant expression if the condition expression which in
  switch statement contains errors.
  (`#63453 <https://github.com/llvm/llvm-project/issues/63453>_`)
- Fixed false positive error diagnostic when pack expansion appears in template
  parameters of a member expression.
  (`#48731 <https://github.com/llvm/llvm-project/issues/48731>`_)
- Fix the contains-errors bit not being set for DeclRefExpr that refers to a
  VarDecl with invalid initializer. This fixes:
  (`#50236 <https://github.com/llvm/llvm-project/issues/50236>`_),
  (`#50243 <https://github.com/llvm/llvm-project/issues/50243>`_),
  (`#48636 <https://github.com/llvm/llvm-project/issues/48636>`_),
  (`#50320 <https://github.com/llvm/llvm-project/issues/50320>`_).
- Fix an assertion when using ``\u0024`` (``$``) as an identifier, by disallowing
  that construct (`#62133 <https://github.com/llvm/llvm-project/issues/38717>_`).
- Fix crash caused by PseudoObjectExprBitfields: NumSubExprs overflow.
  (`#63169 <https://github.com/llvm/llvm-project/issues/63169>_`)
- Fix crash when casting an object to an array type.
  (`#63758 <https://github.com/llvm/llvm-project/issues/63758>_`)
- Fixed false positive error diagnostic observed from mixing ``asm goto`` with
  ``__attribute__((cleanup()))`` variables falsely warning that jumps to
  non-targets would skip cleanup.
- Correcly diagnose jumps into statement expressions.
  This ensures the behavior of Clang is consistent with GCC.
  (`#63682 <https://github.com/llvm/llvm-project/issues/63682>`_)
- Invalidate BlockDecl with implicit return type, in case any of the return
  value exprs is invalid. Propagating the error info up by replacing BlockExpr
  with a RecoveryExpr. This fixes:
  (`#63863 <https://github.com/llvm/llvm-project/issues/63863>_`)
- Invalidate BlockDecl with invalid ParmVarDecl. Remove redundant dump of
  BlockDecl's ParmVarDecl
  (`#64005 <https://github.com/llvm/llvm-project/issues/64005>_`)
- Fix crash on nested templated class with template function call.
  (`#61159 <https://github.com/llvm/llvm-project/issues/61159>_`)
- Fix a hang on valid C code passing a function type as an argument to
  ``typeof`` to form a function declaration.
  (`#64713 <https://github.com/llvm/llvm-project/issues/64713>_`)
- Fixed an issue where accesses to the local variables of a coroutine during
  ``await_suspend`` could be misoptimized, including accesses to the awaiter
  object itself.
  (`#56301 <https://github.com/llvm/llvm-project/issues/56301>`_)
  The current solution may bring performance regressions if the awaiters have
  non-static data members. See
  `#64945 <https://github.com/llvm/llvm-project/issues/64945>`_ for details.

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed a bug where attribute annotations on type specifiers (enums, classes,
  structs, unions, and scoped enums) were not properly ignored, resulting in
  misleading warning messages. Now, such attribute annotations are correctly
  ignored. (`#61660 <https://github.com/llvm/llvm-project/issues/61660>`_)
- GNU attributes preceding C++ style attributes on templates were not properly
  handled, resulting in compilation error. This has been corrected to match the
  behavior exhibited by GCC, which permits mixed ordering of GNU and C++
  attributes.

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
  regarding overloaded `operator[]` with more than one parameter or for static
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
- Fix handling of constexpr dynamic memory allocations in template
  arguments. (`#62462 <https://github.com/llvm/llvm-project/issues/62462>`_)
- Some predefined expressions are now treated as string literals in MSVC
  compatibility mode.
  (`#114 <https://github.com/llvm/llvm-project/issues/114>`_)
- Fix parsing of `auto(x)`, when it is surrounded by parentheses.
  (`#62494 <https://github.com/llvm/llvm-project/issues/62494>`_)
- Fix handling of generic lambda used as template arguments.
  (`#62611 <https://github.com/llvm/llvm-project/issues/62611>`_)
- Allow omitting ``typename`` in the parameter declaration of a friend
  constructor declaration.
  (`#63119 <https://github.com/llvm/llvm-project/issues/63119>`_)
- Fix access of a friend class declared in a local class. Clang previously
  emitted an error when a friend of a local class tried to access it's
  private data members.
- Allow abstract parameter and return types in functions that are
  either deleted or not defined.
  (`#63012 <https://github.com/llvm/llvm-project/issues/63012>`_)
- Fix handling of using-declarations in the init statements of for
  loop declarations.
  (`#63627 <https://github.com/llvm/llvm-project/issues/63627>`_)
- Fix crash when emitting diagnostic for out of order designated initializers
  in C++.
  (`#63605 <https://github.com/llvm/llvm-project/issues/63605>`_)
- Fix crash when using standard C++ modules with OpenMP.
  (`#62359 <https://github.com/llvm/llvm-project/issues/62359>`_)
- Fix crash when using consteval non static data member initialization in
  standard C++ modules.
  (`#60275 <https://github.com/llvm/llvm-project/issues/60275>`_)
- Fix handling of ADL for dependent expressions in standard C++ modules.
  (`#60488 <https://github.com/llvm/llvm-project/issues/60488>`_)
- Fix crash when combining `-ftime-trace` within standard C++ modules.
  (`#60544 <https://github.com/llvm/llvm-project/issues/60544>`_)
- Don't generate template specializations when importing standard C++ modules.
  (`#60693 <https://github.com/llvm/llvm-project/issues/60693>`_)
- Fix the visibility of `initializer list` in the importer of standard C++
  modules. This addresses
  (`#60775 <https://github.com/llvm/llvm-project/issues/60775>`_)
- Allow the use of constrained friend in standard C++ modules.
  (`#60890 <https://github.com/llvm/llvm-project/issues/60890>`_)
- Don't evaluate initializer of used variables in every importer of standard
  C++ modules.
  (`#61040 <https://github.com/llvm/llvm-project/issues/61040>`_)
- Fix the issue that the default `operator==` in standard C++ modules will
  cause duplicate symbol linker error.
  (`#61067 <https://github.com/llvm/llvm-project/issues/61067>`_)
- Fix the false positive ODR check for template names. This addresses the issue
  that we can't include `<ranges>` in multiple module units.
  (`#61317 <https://github.com/llvm/llvm-project/issues/61317>`_)
- Fix crash for inconsistent exported declarations in standard C++ modules.
  (`#61321 <https://github.com/llvm/llvm-project/issues/61321>`_)
- Fix ignoring `#pragma comment` and `#pragma detect_mismatch` directives in
  standard C++ modules.
  (`#61733 <https://github.com/llvm/llvm-project/issues/61733>`_)
- Don't generate virtual tables if the class is defined in another module units
  for Itanium ABI.
  (`#61940 <https://github.com/llvm/llvm-project/issues/61940>`_)
- Fix false postive check for constrained satisfaction in standard C++ modules.
  (`#62589 <https://github.com/llvm/llvm-project/issues/62589>`_)
- Serialize the evaluated constant values for variable declarations in standard
  C++ modules.
  (`#62796 <https://github.com/llvm/llvm-project/issues/62796>`_)
- Merge lambdas in require expressions in standard C++ modules.
  (`#63544 <https://github.com/llvm/llvm-project/issues/63544>`_)

- Fix location of default member initialization in parenthesized aggregate
  initialization.
  (`#63903 <https://github.com/llvm/llvm-project/issues/63903>`_)
- Fix constraint checking of non-generic lambdas.
  (`#63181 <https://github.com/llvm/llvm-project/issues/63181>`_)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

- Preserve ``namespace`` definitions that follow malformed declarations.

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
- A new option ``-mprintf-kind`` has been introduced that controls printf lowering
  scheme. It is currently supported only for HIP and takes following values,
  ``hostcall`` - printing happens during kernel execution via series of hostcalls,
  The scheme requires the system to support pcie atomics.(default)
  ``buffered`` - Scheme uses a debug buffer to populate printf varargs, does not
  rely on pcie atomics support.

X86 Support
^^^^^^^^^^^

- Add ISA of ``AMX-COMPLEX`` which supports ``tcmmimfp16ps`` and
  ``tcmmrlfp16ps``.
- Support ISA of ``SHA512``.
  * Support intrinsic of ``_mm256_sha512msg1_epi64``.
  * Support intrinsic of ``_mm256_sha512msg2_epi64``.
  * Support intrinsic of ``_mm256_sha512rnds2_epi64``.
- Support ISA of ``SM3``.
  * Support intrinsic of ``_mm_sm3msg1_epi32``.
  * Support intrinsic of ``_mm_sm3msg2_epi32``.
  * Support intrinsic of ``_mm_sm3rnds2_epi32``.
- Support ISA of ``SM4``.
  * Support intrinsic of ``_mm(256)_sm4key4_epi32``.
  * Support intrinsic of ``_mm(256)_sm4rnds4_epi32``.
- Support ISA of ``AVX-VNNI-INT16``.
  * Support intrinsic of ``_mm(256)_dpwsud(s)_epi32``.
  * Support intrinsic of ``_mm(256)_dpwusd(s)_epi32``.
  * Support intrinsic of ``_mm(256)_dpwuud(s)_epi32``.
- ``-march=graniterapids-d`` is now supported.

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

- The hard-float ABI is now available in Armv8.1-M configurations that
  have integer MVE instructions (and therefore have FP registers) but
  no scalar or vector floating point computation. Previously, trying
  to select the hard-float ABI on such a target (via
  ``-mfloat-abi=hard`` or a triple ending in ``hf``) would silently
  use the soft-float ABI instead.

- Clang now emits ``-Wunsupported-abi`` if the hard-float ABI is specified
  and the selected processor lacks floating point registers.
  (`#55755 <https://github.com/llvm/llvm-project/issues/55755>`_)

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

- Clang now warns if invalid target triples ``--target=aarch64-*-eabi`` or
  ``--target=arm-*-elf`` are specified.

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

- Patchable function entry (``-fpatchable-function-entry``) is now supported
  on LoongArch.
- An ABI mismatch between GCC and Clang related to the handling of empty structs
  in C++ parameter passing under ``lp64d`` ABI was fixed.
- Unaligned memory accesses can be toggled by ``-m[no-]unaligned-access`` or the
  aliases ``-m[no-]strict-align``.
- Non ``$``-prefixed GPR names (e.g. ``r4`` and ``a0``) are allowed in inlineasm
  like GCC does.
- The ``-march=native`` ``-mtune=`` options and ``__loongarch_{arch,tune}``
  macros are now supported.

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
- The rules for ordering of extensions in ``-march`` strings were relaxed. A
  canonical ordering is no longer enforced on ``z*``, ``s*``, and ``x*``
  prefixed extensions.
- An ABI mismatch between GCC and Clang related to the handling of empty
  structs in C++ parameter passing under the hard floating point calling
  conventions was fixed.
- Support the RVV intrinsics v0.12. Please checkout `the RVV C intrinsics
  specification
  <https://github.com/riscv-non-isa/rvv-intrinsic-doc/releases/tag/v0.12.0>`_.
  It is expected there won't be any incompatibility from this v0.12 to the
  specifications planned for v1.0.

  * Added vector intrinsics that models control to the rounding mode
    (``frm`` and ``vxrm``) for the floating-point instruction intrinsics and the 
    fixed-point instruction intrinsics.
  * Added intrinsics for reinterpret cast between vector boolean and vector
    integer ``m1`` value
  * Removed the ``vread_csr`` and ``vwrite_csr`` intrinsics
- Default ``-fdebug-dwarf-version=`` is downgraded to 4 to work around
  incorrect DWARF related to ULEB128 and linker compatibility before
  ``R_RISCV_SET_ULEB128`` becomes more widely supported.
  (`D157663 <https://reviews.llvm.org/D157663>`_).

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^
- Clang has been updated to align its default language standard for CUDA/HIP with
  that of C++. The standard has now been enhanced to gnu++17, supplanting the
  previously used c++14.

CUDA Support
^^^^^^^^^^^^
- Clang now supports CUDA SDK up to 12.1

AIX Support
^^^^^^^^^^^
- Add an AIX-only link-time option, `-mxcoff-build-id=0xHEXSTRING`, to allow users
  to embed a hex id in their binary such that it's readable by the program itself.
  This option is an alternative to the `--build-id=0xHEXSTRING` GNU linker option
  which is currently not supported by the AIX linker.

- Introduced the ``-mxcoff-roptr`` option to place constant objects with
  relocatable address values in the read-only data section. This option should
  be used with the ``-fdata-sections`` option, and is not supported with
  ``-fno-data-sections``. When ``-mxcoff-roptr`` is in effect at link time,
  read-only data sections with relocatable address values that resolve to
  imported symbols are made writable.

WebAssembly Support
^^^^^^^^^^^^^^^^^^^
- Shared library support (and PIC code generation) for WebAssembly is no longer
  limited to the Emscripten target OS and now works with other targets such as
  wasm32-wasi.  Note that the `format
  <https://github.com/WebAssembly/tool-conventions/blob/main/DynamicLinking.md>`_
  is not yet stable and may change between LLVM versions.  Also, WASI does not
  yet have facilities to load dynamic libraries.

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
- Add ``__builtin_elementwise_pow`` builtin for floating point types only.

AST Matchers
------------

- Add ``coroutineBodyStmt`` matcher.

- The ``hasBody`` matcher now matches coroutine body nodes in
  ``CoroutineBodyStmts``.

- Add ``arrayInitIndexExpr`` and ``arrayInitLoopExpr`` matchers.

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
- Add ``KeepEmptyLinesAtEOF`` to keep empty lines at end of file.
- Add ``RemoveParentheses`` to remove redundant parentheses.
- Add ``TypeNames`` to treat listed non-keyword identifiers as type names.
- Add ``AlignConsecutiveShortCaseStatements`` which can be used to align case
  labels in conjunction with ``AllowShortCaseLabelsOnASingleLine``.
- Add ``SpacesInParens`` style with ``SpacesInParensOptions`` to replace
  ``SpacesInConditionalStatement``, ``SpacesInCStyleCastParentheses``,
  ``SpaceInEmptyParentheses``, and ``SpacesInParentheses``.

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
  bit-field whose width depends on a template parameter.

- Added ``CXBinaryOperatorKind`` and ``CXUnaryOperatorKind``.
  (`#29138 <https://github.com/llvm/llvm-project/issues/29138>`_)

Static Analyzer
---------------

- Fix incorrect alignment attribute on the this parameter of certain
  non-complete destructors when using the Microsoft ABI.
  (`#60465 <https://github.com/llvm/llvm-project/issues/60465>`_)

- Removed the deprecated
  ``consider-single-element-arrays-as-flexible-array-members`` analyzer option.
  Any use of this flag will result in an error.
  Use `-fstrict-flex-arrays=<n>
  <https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fstrict-flex-arrays>`_

- Better modeling of lifetime-extended memory regions. As a result, the
  ``MoveChecker`` raises more true-positive reports.

- Fixed some bugs (including crashes) around the handling of constant global
  arrays and their initializer expressions.

- The ``CStringChecker`` will invalidate less if the copy operation is
  inferable to be bounded. For example, if the arguments of ``strcpy`` are
  known to be of certain lengths and that are in-bounds.

   .. code-block:: c++

    struct {
      void *ptr;
      char arr[4];
    } x;
    x.ptr = malloc(1);
    // extent of 'arr' is 4, and writing "hi\n" (4 characters),
    // thus no buffer overflow can happen
    strcpy(x.arr, "hi\n");
    free(x.ptr); // no longer reports memory leak here

  Similarly, functions like ``strsep`` now won't invalidate the object
  containing the destination buffer, because it can never overflow.
  Note that, ``std::copy`` is still not modeled, and as such, it will still
  invalidate the enclosing object on call.
  (`#55019 <https://github.com/llvm/llvm-project/issues/55019>`_)

- Implement ``BufferOverlap`` check for ``sprint``/``snprintf``
  The ``CStringChecker`` checks for buffer overlaps for ``sprintf`` and
  ``snprintf``.

- Objective-C support was improved around checking ``_Nonnull`` and
  ``_Nullable`` including block pointers and literal objects.

- Let the ``StreamChecker`` detect ``NULL`` streams instead of by
  ``StdCLibraryFunctions``.
  ``StreamChecker`` improved on the ``fseek`` modeling for the ``SEEK_SET``,
  ``SEEK_END``, ``SEEK_CUR`` arguments.

- ``StdCLibraryFunctionArgs`` was merged into the ``StdCLibraryFunctions``.
  The diagnostics of the ``StdCLibraryFunctions`` was improved.

- ``QTimer::singleShot`` now doesn't raise false-positives for memory leaks by
  the ``MallocChecker``.
  (`#39713 <https://github.com/llvm/llvm-project/issues/39713>`_)

- Fixed the infamous unsigned index false-positives in the
  ``ArrayBoundCheckerV2`` checker.
  (`#44493 <https://github.com/llvm/llvm-project/issues/44493>`_)

- Now, taint propagations are tracked further back until the real taint source.
  This improves all taint-related diagnostics.

- Fixed a null-pointer dereference crash inside the ``MoveChecker``.

.. _release-notes-sanitizers:

Sanitizers
----------
- Several more sanitizers are now ported to LoongArch: MSan, DFsan, Profile, XRay and libFuzzer.

Python Binding Changes
----------------------
The following methods have been added:

- ``clang_Location_isInSystemHeader`` exposed via the ``is_in_system_header``
  property of the `Location` class.

Configurable Multilib
---------------------
The BareMetal toolchain for AArch64 & ARM now supports multilib, configurable
via ``multilib.yaml``. See `the multilib docs <https://clang.llvm.org/docs/Multilib.html>`_
for more details.

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
