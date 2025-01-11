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

- The ``le32`` and ``le64`` targets have been removed.

- ``clang -m32`` defaults to ``-mcpu=v9`` on SPARC Linux now.  Distros
  still supporting SPARC V8 CPUs need to specify ``-mcpu=v8`` with a
  `config file
  <https://clang.llvm.org/docs/UsersManual.html#configuration-files>`_.

- The ``clang-rename`` tool has been removed.

- Removed support for RenderScript targets. This technology is
  `officially deprecated <https://developer.android.com/guide/topics/renderscript/compute>`_
  and users are encouraged to
  `migrate to Vulkan <https://developer.android.com/guide/topics/renderscript/migrate>`_
  or other options.

- Clang now emits distinct type-based alias analysis tags for incompatible
  pointers by default, enabling more powerful alias analysis when accessing
  pointer types. This change may silently change code behavior for code
  containing strict-aliasing violations. The new default behavior can be
  disabled using ``-fno-pointer-tbaa``.

C/C++ Language Potentially Breaking Changes
-------------------------------------------

- Clang now rejects ``_Complex _BitInt`` types.

C++ Specific Potentially Breaking Changes
-----------------------------------------

- The type trait builtin ``__is_nullptr`` has been removed, since it has very
  few users and can be written as ``__is_same(__remove_cv(T), decltype(nullptr))``,
  which GCC supports as well.

- Clang will now correctly diagnose as ill-formed a constant expression where an
  enum without a fixed underlying type is set to a value outside the range of
  the enumeration's values.

  .. code-block:: c++

    enum E { Zero, One, Two, Three, Four };
    constexpr E Val1 = (E)3;  // Ok
    constexpr E Val2 = (E)7;  // Ok
    constexpr E Val3 = (E)8;  // Now ill-formed, out of the range [0, 7]
    constexpr E Val4 = (E)-1; // Now ill-formed, out of the range [0, 7]

  Since Clang 16, it has been possible to suppress the diagnostic via
  `-Wno-enum-constexpr-conversion`, to allow for a transition period for users.
  Now, in Clang 20, **it is no longer possible to suppress the diagnostic**.

- Extraneous template headers are now ill-formed by default.
  This error can be disable with ``-Wno-error=extraneous-template-head``.

  .. code-block:: c++

    template <> // error: extraneous template head
    template <typename T>
    void f();

- During constant evaluation, comparisons between different evaluations of the
  same string literal are now correctly treated as non-constant, and comparisons
  between string literals that cannot possibly overlap in memory are now treated
  as constant. This updates Clang to match the anticipated direction of open core
  issue `CWG2765 <http://wg21.link/CWG2765>`, but is subject to change once that
  issue is resolved.

  .. code-block:: c++

    constexpr const char *f() { return "hello"; }
    constexpr const char *g() { return "world"; }
    // Used to evaluate to false, now error: non-constant comparison.
    constexpr bool a = f() == f();
    // Might evaluate to true or false, as before.
    bool at_runtime() { return f() == f(); }
    // Was error, now evaluates to false.
    constexpr bool b = f() == g();

- Clang will now correctly not consider pointers to non classes for covariance
  and disallow changing return type to a type that doesn't have the same or less cv-qualifications.

  .. code-block:: c++

    struct A {
      virtual const int *f() const;
      virtual const std::string *g() const;
    };
    struct B : A {
      // Return type has less cv-qualification but doesn't point to a class.
      // Error will be generated.
      int *f() const override;

      // Return type doesn't have more cv-qualification also not the same or
      // less cv-qualification.
      // Error will be generated.
      volatile std::string *g() const override;
    };

- The warning ``-Wdeprecated-literal-operator`` is now on by default, as this is
  something that WG21 has shown interest in removing from the language. The
  result is that anyone who is compiling with ``-Werror`` should see this
  diagnostic.  To fix this diagnostic, simply removing the space character from
  between the ``operator""`` and the user defined literal name will make the
  source no longer deprecated. This is consistent with `CWG2521 <https://cplusplus.github.io/CWG/issues/2521.html>_`.

  .. code-block:: c++

    // Now diagnoses by default.
    unsigned operator"" _udl_name(unsigned long long);
    // Fixed version:
    unsigned operator""_udl_name(unsigned long long);

- Clang will now produce an error diagnostic when ``[[clang::lifetimebound]]`` is
  applied on a parameter or an implicit object parameter of a function that
  returns void. This was previously ignored and had no effect. (#GH107556)

  .. code-block:: c++

    // Now diagnoses with an error.
    void f(int& i [[clang::lifetimebound]]);

- Clang will now produce an error diagnostic when ``[[clang::lifetimebound]]``
  is applied on a type (instead of a function parameter or an implicit object
  parameter); this includes the case when the attribute is specified for an
  unnamed function parameter. These were previously ignored and had no effect.
  (#GH118281)

  .. code-block:: c++

    // Now diagnoses with an error.
    int* [[clang::lifetimebound]] x;
    // Now diagnoses with an error.
    void f(int* [[clang::lifetimebound]] i);
    // Now diagnoses with an error.
    void g(int* [[clang::lifetimebound]]);

- Clang now rejects all field accesses on null pointers in constant expressions. The following code
  used to work but will now be rejected:

  .. code-block:: c++

    struct S { int a; int b; };
    constexpr const int *p = &((S*)nullptr)->b;

  Previously, this code was erroneously accepted.

- Clang will now consider the implicitly deleted destructor of a union or
  a non-union class without virtual base class to be ``constexpr`` in C++20
  mode (Clang 19 only did so in C++23 mode but the standard specification for
  this changed in C++20). (#GH85550)

  .. code-block:: c++

    struct NonLiteral {
      NonLiteral() {}
      ~NonLiteral() {}
    };

    template <class T>
    struct Opt {
      union {
        char c;
        T data;
      };
      bool engaged = false;

      constexpr Opt() {}
      constexpr ~Opt() {
        if (engaged)
          data.~T();
      }
    };

    // Previously only accepted in C++23 and later, now also accepted in C++20.
    consteval void foo() { Opt<NonLiteral>{}; }

ABI Changes in This Version
---------------------------

- Fixed Microsoft name mangling of placeholder, auto and decltype(auto), return types for MSVC 1920+. This change resolves incompatibilities with code compiled by MSVC 1920+ but will introduce incompatibilities with code compiled by earlier versions of Clang unless such code is built with the compiler option -fms-compatibility-version=19.14 to imitate the MSVC 1914 mangling behavior.
- Fixed the Itanium mangling of the construction vtable name. This change will introduce incompatibilities with code compiled by Clang 19 and earlier versions, unless the -fclang-abi-compat=19 option is used. (#GH108015)
- Mangle member-like friend function templates as members of the enclosing class. (#GH110247, #GH110503)

AST Dumping Potentially Breaking Changes
----------------------------------------

Clang Frontend Potentially Breaking Changes
-------------------------------------------

Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------
- Parts of the interface returning string results will now return
  the empty string ``""`` when no result is available, instead of ``None``.
- Calling a property on the ``CompletionChunk`` or ``CompletionString`` class
  statically now leads to an error, instead of returning a ``CachedProperty`` object
  that is used internally. Properties are only available on instances.
- For a single-line ``SourceRange`` and a ``SourceLocation`` in the same line,
  but after the end of the ``SourceRange``, ``SourceRange.__contains__``
  used to incorrectly return ``True``. (#GH22617), (#GH52827)

What's New in Clang |release|?
==============================
Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

C++ Language Changes
--------------------
- Allow single element access of GCC vector/ext_vector_type object to be
  constant expression. Supports the `V.xyzw` syntax and other tidbits
  as seen in OpenCL. Selecting multiple elements is left as a future work.
- Implement `CWG1815 <https://wg21.link/CWG1815>`_. Support lifetime extension
  of temporary created by aggregate initialization using a default member
  initializer.

- Accept C++26 user-defined ``static_assert`` messages in C++11 as an extension.

- Add ``__builtin_elementwise_popcount`` builtin for integer types only.

- Add ``__builtin_elementwise_fmod`` builtin for floating point types only.

- Add ``__builtin_elementwise_minimum`` and ``__builtin_elementwise_maximum``
  builtin for floating point types only.

- The builtin type alias ``__builtin_common_type`` has been added to improve the
  performance of ``std::common_type``.

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Add ``__builtin_is_implicit_lifetime`` intrinsic, which supports
  `P2647R1 A trait for implicit lifetime types <https://wg21.link/p2674r1>`_

- Add ``__builtin_is_virtual_base_of`` intrinsic, which supports
  `P2985R0 A type trait for detecting virtual base classes <https://wg21.link/p2985r0>`_

- Implemented `P2893R3 Variadic Friends <https://wg21.link/P2893>`_

- Implemented `P2747R2 constexpr placement new <https://wg21.link/P2747R2>`_.

- Added the ``__builtin_is_within_lifetime`` builtin, which supports
  `P2641R4 Checking if a union alternative is active <https://wg21.link/p2641r4>`_

- Implemented `P3176R1 The Oxford variadic comma <https://wg21.link/P3176R1>`_

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Removed the restriction to literal types in constexpr functions in C++23 mode.

- Extend lifetime of temporaries in mem-default-init for P2718R0. Clang now fully
  supports `P2718R0 Lifetime extension in range-based for loops <https://wg21.link/P2718R0>`_.
  
- ``__cpp_explicit_this_parameter`` is now defined. (#GH82780)

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^


Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Allow calling initializer list constructors from initializer lists with
  a single element of the same type instead of always copying.
  (`CWG2137: List-initialization from object of same type <https://cplusplus.github.io/CWG/issues/2137.html>`)

- Speculative resolution for CWG2311 implemented so that the implementation of CWG2137 doesn't remove
  previous cases where guaranteed copy elision was done. Given a prvalue ``e`` of class type
  ``T``, ``T{e}`` will try to resolve an initializer list constructor and will use it if successful.
  Otherwise, if there is no initializer list constructor, the copy will be elided as if it was ``T(e)``.
  (`CWG2311: Missed case for guaranteed copy elision <https://cplusplus.github.io/CWG/issues/2311.html>`)

- Casts from a bit-field to an integral type is now not considered narrowing if the
  width of the bit-field means that all potential values are in the range
  of the target type, even if the type of the bit-field is larger.
  (`CWG2627: Bit-fields and narrowing conversions <https://cplusplus.github.io/CWG/issues/2627.html>`_)

- ``nullptr`` is now promoted to ``void*`` when passed to a C-style variadic function.
  (`CWG722: Can nullptr be passed to an ellipsis? <https://cplusplus.github.io/CWG/issues/722.html>`_)

- Allow ``void{}`` as a prvalue of type ``void``.
  (`CWG2351: void{} <https://cplusplus.github.io/CWG/issues/2351.html>`_).

- Clang now has improved resolution to CWG2398, allowing class templates to have
  default arguments deduced when partial ordering.

- Clang now allows comparing unequal object pointers that have been cast to ``void *``
  in constant expressions. These comparisons always worked in non-constant expressions.
  (`CWG2749: Treatment of "pointer to void" for relational comparisons <https://cplusplus.github.io/CWG/issues/2749.html>`_).

- Reject explicit object parameters with type ``void`` (``this void``).
  (`CWG2915: Explicit object parameters of type void <https://cplusplus.github.io/CWG/issues/2915.html>`_).

- Clang now allows trailing requires clause on explicit deduction guides.
  (`CWG2707: Deduction guides cannot have a trailing requires-clause <https://cplusplus.github.io/CWG/issues/2707.html>`_).

- Clang now diagnoses a space in the first production of a ``literal-operator-id``
  by default.
  (`CWG2521: User-defined literals and reserved identifiers <https://cplusplus.github.io/CWG/issues/2521.html>`_).

- Fix name lookup for a dependent base class that is the current instantiation.
  (`CWG591: When a dependent base class is the current instantiation <https://cplusplus.github.io/CWG/issues/591.html>`_).

- Clang now allows calling explicit object member functions directly with prvalues
  instead of always materializing a temporary, meaning by-value explicit object parameters
  do not need to move from a temporary.
  (`CWG2813: Class member access with prvalues <https://cplusplus.github.io/CWG/issues/2813.html>`_).

C Language Changes
------------------

- Extend clang's ``<limits.h>`` to define ``LONG_LONG_*`` macros for Android's bionic.

C2y Feature Support
^^^^^^^^^^^^^^^^^^^

- Updated conformance for `N3298 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3298.htm>`_
  which adds the ``i`` and ``j`` suffixes for the creation of a ``_Complex``
  constant value. Clang has always supported these suffixes as a GNU extension,
  so ``-Wgnu-imaginary-constant`` no longer has effect in C modes, as this is
  now a C2y extension in C. ``-Wgnu-imaginary-constant`` still applies in C++
  modes.

- Clang updated conformance for `N3370 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3370.htm>`_
  case range expressions. This feature was previously supported by Clang as a
  GNU extension, so ``-Wgnu-case-range`` no longer has effect in C modes, as
  this is now a C2y extension in C. ``-Wgnu-case-range`` still applies in C++
  modes.

- Clang implemented support for `N3344 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3344.pdf>`_
  which disallows a ``void`` parameter from having a qualifier or storage class
  specifier. Note that ``register void`` was previously accepted in all C
  language modes but is now rejected (all of the other qualifiers and storage
  class specifiers were previously rejected).

- Updated conformance for `N3364 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3364.pdf>`_
  on floating-point translation-time initialization with signaling NaN. This
  paper adopts Clang's existing practice, so there were no changes to compiler
  behavior.

- Implemented support for `N3341 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3341.pdf>`_
  which makes empty structure and union objects implementation-defined in C.
  ``-Wgnu-empty-struct`` will be emitted in C23 and earlier modes because the
  behavior is a conforming GNU extension in those modes, but will no longer
  have an effect in C2y mode.

- Updated conformance for `N3342 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3342.pdf>`_
  which made qualified function types implementation-defined rather than
  undefined. Clang has always accepted ``const`` and ``volatile`` qualified
  function types by ignoring the qualifiers.

- Updated conformance for `N3346 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3346.pdf>`_
  which changes some undefined behavior around initialization to instead be
  constraint violations. This paper adopts Clang's existing practice, so there
  were no changes to compiler behavior.

C23 Feature Support
^^^^^^^^^^^^^^^^^^^

- Clang now supports `N3029 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3029.htm>`_ Improved Normal Enumerations.
- Clang now officially supports `N3030 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3030.htm>`_ Enhancements to Enumerations. Clang already supported it as an extension, so there were no changes to compiler behavior.
- Fixed the value of ``BOOL_WIDTH`` in ``<limits.h>`` to return ``1``
  explicitly, as mandated by the standard. Fixes #GH117348

Non-comprehensive list of changes in this release
-------------------------------------------------

- The floating point comparison builtins (``__builtin_isgreater``,
  ``__builtin_isgreaterequal``, ``__builtin_isless``, etc.) and
  ``__builtin_signbit`` can now be used in constant expressions.
- Plugins can now define custom attributes that apply to statements
  as well as declarations.
- ``__builtin_abs`` function can now be used in constant expressions.

- The new builtin ``__builtin_counted_by_ref`` was added. In contexts where the
  programmer needs access to the ``counted_by`` attribute's field, but it's not
  available --- e.g. in macros. For instance, it can be used to automatically
  set the counter during allocation in the Linux kernel:

  .. code-block:: c

     /* A simplified version of Linux allocation macros */
     #define alloc(PTR, FAM, COUNT) ({ \
         sizeof_t __ignored_assignment;                             \
         typeof(P) __p;                                             \
         size_t __size = sizeof(*P) + sizeof(*P->FAM) * COUNT;      \
         __p = malloc(__size);                                      \
         *_Generic(                                                 \
           __builtin_counted_by_ref(__p->FAM),                      \
             void *: &__ignored_assignment,                         \
             default: __builtin_counted_by_ref(__p->FAM)) = COUNT;  \
         __p;                                                       \
     })

  The flexible array member (FAM) can now be accessed immediately without causing
  issues with the sanitizer because the counter is automatically set.

- The following builtins can now be used in constant expressions: ``__builtin_reduce_add``,
  ``__builtin_reduce_mul``, ``__builtin_reduce_and``, ``__builtin_reduce_or``,
  ``__builtin_reduce_xor``, ``__builtin_elementwise_popcount``,
  ``__builtin_elementwise_bitreverse``, ``__builtin_elementwise_add_sat``,
  ``__builtin_elementwise_sub_sat``, ``__builtin_reduce_min`` (For integral element type),
  ``__builtin_reduce_max`` (For integral element type).

- Clang now rejects ``_BitInt`` matrix element types if the bit width is less than ``CHAR_WIDTH`` or
  not a power of two, matching preexisting behaviour for vector types.

- Matrix types (a Clang extension) can now be used in pseudo-destructor expressions,
  which allows them to be stored in STL containers.

- In the ``-ftime-report`` output, the new "Clang time report" group replaces
  the old "Clang front-end time report" and includes "Front end", "LLVM IR
  generation", "Optimizer", and "Machine code generation".

New Compiler Flags
------------------

- The ``-fc++-static-destructors={all,thread-local,none}`` flag was
  added to control which C++ variables have static destructors
  registered: all (the default) does so for all variables, thread-local
  only for thread-local variables, and none (which corresponds to the
  existing ``-fno-c++-static-destructors`` flag) skips all static
  destructors registration.

- The ``-Warray-compare`` warning has been added to warn about array comparison
  on versions older than C++20.

- The ``-Warray-compare-cxx26`` warning has been added to warn about array comparison
  starting from C++26, this warning is enabled as an error by default.

- clang-cl and clang-dxc now support ``-fdiagnostics-color=[auto|never|always]``
  in addition to ``-f[no-]color-diagnostics``.

Deprecated Compiler Flags
-------------------------

- ``-fheinous-gnu-extensions`` is deprecated; it is now equivalent to
  specifying ``-Wno-error=invalid-gnu-asm-cast`` and may be removed in the
  future.

Modified Compiler Flags
-----------------------

- The ``-ffp-model`` option has been updated to enable a more limited set of
  optimizations when the ``fast`` argument is used and to accept a new argument,
  ``aggressive``. The behavior of ``-ffp-model=aggressive`` is equivalent
  to the previous behavior of ``-ffp-model=fast``. The updated
  ``-ffp-model=fast`` behavior no longer assumes finite math only and uses
  the ``promoted`` algorithm for complex division when possible rather than the
  less basic (limited range) algorithm.

- The ``-fveclib`` option has been updated to enable ``-fno-math-errno`` for
  ``-fveclib=ArmPL`` and ``-fveclib=SLEEF``. This gives Clang more opportunities
  to utilize these vector libraries. The behavior for all other vector function
  libraries remains unchanged.

- The ``-Wnontrivial-memcall`` warning has been added to warn about
  passing non-trivially-copyable destination parameter to ``memcpy``,
  ``memset`` and similar functions for which it is a documented undefined
  behavior. It is implied by ``-Wnontrivial-memaccess``

- Added ``-fmodules-reduced-bmi`` flag corresponding to
  ``-fexperimental-modules-reduced-bmi`` flag. The ``-fmodules-reduced-bmi`` flag
  is intended to be enabled by default in the future.

Removed Compiler Flags
-------------------------

- The compiler flag `-Wenum-constexpr-conversion` (and the `Wno-`, `Wno-error-`
  derivatives) is now removed, since it's no longer possible to suppress the
  diagnostic (see above). Users can expect an `unknown warning` diagnostic if
  it's still in use.

Attribute Changes in Clang
--------------------------

- The ``swift_attr`` can now be applied to types. To make it possible to use imported APIs
  in Swift safely there has to be a way to annotate individual parameters and result types
  with relevant attributes that indicate that e.g. a block is called on a particular actor
  or it accepts a Sendable or global-actor (i.e. ``@MainActor``) isolated parameter.

  For example:

  .. code-block:: objc

     @interface MyService
       -(void) handle: (void (^ __attribute__((swift_attr("@Sendable"))))(id)) handler;
     @end

- Clang now disallows more than one ``__attribute__((ownership_returns(class, idx)))`` with
  different class names attached to one function.

- Introduced a new format attribute ``__attribute__((format(syslog, 1, 2)))`` from OpenBSD.

- The ``hybrid_patchable`` attribute is now supported on ARM64EC targets. It can be used to specify
  that a function requires an additional x86-64 thunk, which may be patched at runtime.

- The attribute ``[[clang::no_specializations]]`` has been added to warn
  users that a specific template shouldn't be specialized. This is useful for
  e.g. standard library type traits, where adding a specialization results in
  undefined behaviour.

- ``[[clang::lifetimebound]]`` is now explicitly disallowed on explicit object member functions
  where they were previously silently ignored.

- Clang now automatically adds ``[[clang::lifetimebound]]`` to the parameters of
  ``std::span, std::string_view`` constructors, this enables Clang to capture
  more cases where the returned reference outlives the object.
  (#GH100567)

- Clang now correctly diagnoses the use of ``btf_type_tag`` in C++ and ignores
  it; this attribute is a C-only attribute, and caused crashes with template
  instantiation by accidentally allowing it in C++ in some circumstances.
  (#GH106864)

- Introduced a new attribute ``[[clang::coro_await_elidable]]`` on coroutine return types
  to express elideability at call sites where the coroutine is invoked under a safe elide context.

- Introduced a new attribute ``[[clang::coro_await_elidable_argument]]`` on function parameters
  to propagate safe elide context to arguments if such function is also under a safe elide context.

- The documentation of the ``[[clang::musttail]]`` attribute was updated to
  note that the lifetimes of all local variables end before the call. This does
  not change the behaviour of the compiler, as this was true for previous
  versions.

- Fix a bug where clang doesn't automatically apply the ``[[gsl::Owner]]`` or
  ``[[gsl::Pointer]]`` to STL explicit template specialization decls. (#GH109442)

- Clang now supports ``[[clang::lifetime_capture_by(X)]]``. Similar to lifetimebound, this can be
  used to specify when a reference to a function parameter is captured by another capturing entity ``X``.

- The ``target_version`` attribute is now only supported for AArch64 and RISC-V architectures.

- Clang now permits the usage of the placement new operator in ``[[msvc::constexpr]]``
  context outside of the std namespace. (#GH74924)

- Clang now disallows the use of attributes after the namespace name. (#GH121407)

Improvements to Clang's diagnostics
-----------------------------------

- Some template related diagnostics have been improved.

  .. code-block:: c++

     void foo() { template <typename> int i; } // error: templates can only be declared in namespace or class scope

     struct S {
      template <typename> int i; // error: non-static data member 'i' cannot be declared as a template
     };

- Clang now has improved diagnostics for functions with explicit 'this' parameters. Fixes #GH97878

- Clang now diagnoses dangling references to fields of temporary objects. Fixes #GH81589.

- Clang now diagnoses undefined behavior in constant expressions more consistently. This includes invalid shifts, and signed overflow in arithmetic.

- -Wdangling-assignment-gsl is enabled by default.
- Clang now always preserves the template arguments as written used
  to specialize template type aliases.

- Clang now diagnoses the use of ``main`` in an ``extern`` context as invalid according to [basic.start.main] p3. Fixes #GH101512.

- Clang now diagnoses when the result of a [[nodiscard]] function is discarded after being cast in C. Fixes #GH104391.

- Don't emit duplicated dangling diagnostics. (#GH93386).

- Improved diagnostic when trying to befriend a concept. (#GH45182).

- Added the ``-Winvalid-gnu-asm-cast`` diagnostic group to control warnings
  about use of "noop" casts for lvalues (a GNU extension). This diagnostic is
  a warning which defaults to being an error, is enabled by default, and is
  also controlled by the now-deprecated ``-fheinous-gnu-extensions`` flag.

- Added the ``-Wdecls-in-multiple-modules`` option to assist users to identify
  multiple declarations in different modules, which is the major reason of the slow
  compilation speed with modules. This warning is disabled by default and it needs
  to be explicitly enabled or by ``-Weverything``.

- Improved diagnostic when trying to overload a function in an ``extern "C"`` context. (#GH80235)

- Clang now respects lifetimebound attribute for the assignment operator parameter. (#GH106372).

- The lifetimebound and GSL analysis in clang are coherent, allowing clang to
  detect more use-after-free bugs. (#GH100549).

- Clang now diagnoses dangling cases where a gsl-pointer is constructed from a gsl-owner object inside a container (#GH100384).

- Clang now warns for u8 character literals used in C23 with ``-Wpre-c23-compat`` instead of ``-Wpre-c++17-compat``.

- Clang now diagnose when importing module implementation partition units in module interface units.

- Don't emit bogus dangling diagnostics when ``[[gsl::Owner]]`` and `[[clang::lifetimebound]]` are used together (#GH108272).

- The ``-Wreturn-stack-address`` warning now also warns about addresses of
  local variables passed to function calls using the ``[[clang::musttail]]``
  attribute.

- Clang now diagnoses cases where a dangling ``GSLOwner<GSLPointer>`` object is constructed, e.g. ``std::vector<string_view> v = {std::string()};`` (#GH100526).

- Clang now diagnoses when a ``requires`` expression has a local parameter of void type, aligning with the function parameter (#GH109831).

- Clang now emits a diagnostic note at the class declaration when the method definition does not match any declaration (#GH110638).

- Clang now omits warnings for extra parentheses in fold expressions with single expansion (#GH101863).

- The warning for an unsupported type for a named register variable is now phrased ``unsupported type for named register variable``,
  instead of ``bad type for named register variable``. This makes it clear that the type is not supported at all, rather than being
  suboptimal in some way the error fails to mention (#GH111550).

- Clang now emits a ``-Wdepredcated-literal-operator`` diagnostic, even if the
  name was a reserved name, which we improperly allowed to suppress the
  diagnostic.

- Clang now diagnoses ``[[deprecated]]`` attribute usage on local variables (#GH90073).

- Fix false positives when `[[gsl::Owner/Pointer]]` and `[[clang::lifetimebound]]` are used together.

- Improved diagnostic message for ``__builtin_bit_cast`` size mismatch (#GH115870).

- Clang now omits shadow warnings for enum constants in separate class scopes (#GH62588).

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

- Clang now diagnoses ``= delete("reason")`` extension warnings only in pedantic mode rather than on by default. (#GH109311).

- Clang now diagnoses missing return value in functions containing ``if consteval`` (#GH116485).

- Clang now correctly recognises code after a call to a ``[[noreturn]]`` constructor
  as unreachable (#GH63009).

- Clang now omits shadowing warnings for parameter names in explicit object member functions (#GH95707).

- Improved error recovery for function call arguments with trailing commas (#GH100921).

- For an rvalue reference bound to a temporary struct with an integer member, Clang will detect constant integer overflow
  in the initializer for the integer member (#GH46755).

- Fixed a false negative ``-Wunused-private-field`` diagnostic when a defaulted comparison operator is defined out of class (#GH116961).

- Clang now diagnoses dangling references for C++20's parenthesized aggregate initialization (#101957).

- Fixed a bug where Clang would not emit ``-Wunused-private-field`` warnings when an unrelated class 
  defined a defaulted comparison operator (#GH116270).

  .. code-block:: c++

    class A {
    private:
      int a; // warning: private field 'a' is not used, no diagnostic previously
    };

    class C {
      bool operator==(const C&) = default;
    };

- Clang now emits `-Wdangling-capture` diangostic when a STL container captures a dangling reference.

  .. code-block:: c++

    void test() {
      std::vector<std::string_view> views;
      views.push_back(std::string("123")); // warning
    }

- Clang now emits a ``-Wtautological-compare`` diagnostic when a check for
  pointer addition overflow is always true or false, because overflow would
  be undefined behavior.

  .. code-block:: c++

    bool incorrect_overflow_check(const char *ptr, size_t index) {
      return ptr + index < ptr; // warning
    }

- Clang now emits a ``-Wvarargs`` diagnostic when the second argument
  to ``va_arg`` is of array type, which is an undefined behavior (#GH119360).

  .. code-block:: c++

    void test() {
      va_list va;
      va_arg(va, int[10]); // warning
    }

- Fix -Wdangling false positives on conditional operators (#120206).

- Fixed a bug where Clang hung on an unsupported optional scope specifier ``::`` when parsing
  Objective-C. Clang now emits a diagnostic message instead of hanging.

- The :doc:`ThreadSafetyAnalysis` now supports passing scoped capabilities into functions:
  an attribute on the scoped capability parameter indicates both the expected associated capabilities and,
  like in the case of attributes on the function declaration itself, their state before and after the call.

  .. code-block:: c++

    #include "mutex.h"

    Mutex mu1, mu2;
    int a GUARDED_BY(mu1);

    void require(MutexLocker& scope REQUIRES(mu1)) {
      scope.Unlock();
      a = 0; // Warning!  Requires mu1.
      scope.Lock();
    }

    void testParameter() {
      MutexLocker scope(&mu1), scope2(&mu2);
      require(scope2); // Warning! Mutex managed by 'scope2' is 'mu2' instead of 'mu1'
      require(scope); // OK.
      scope.Unlock();
      require(scope); // Warning!  Requires mu1.
    }

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------

- Fixed the definition of ``ATOMIC_FLAG_INIT`` in ``<stdatomic.h>`` so it can
  be used in C++.
- Fixed a failed assertion when checking required literal types in C context. (#GH101304).
- Fixed a crash when trying to transform a dependent address space type. Fixes #GH101685.
- Fixed a crash when diagnosing format strings and encountering an empty
  delimited escape sequence (e.g., ``"\o{}"``). #GH102218
- Fixed a crash using ``__array_rank`` on 64-bit targets. (#GH113044).
- The warning emitted for an unsupported register variable type now points to
  the unsupported type instead of the ``register`` keyword (#GH109776).
- Fixed a crash when emit ctor for global variant with flexible array init (#GH113187).
- Fixed a crash when GNU statement expression contains invalid statement (#GH113468).
- Fixed a crash when passing the variable length array type to ``va_arg`` (#GH119360).
- Fixed a failed assertion when using ``__attribute__((noderef))`` on an
  ``_Atomic``-qualified type (#GH116124).
- No longer incorrectly diagnosing use of a deleted destructor when the
  selected overload of ``operator delete`` for that type is a destroying delete
  (#GH46818).
- No longer return ``false`` for ``noexcept`` expressions involving a
  ``delete`` which resolves to a destroying delete but the type of the object
  being deleted has a potentially throwing destructor (#GH118660).

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fix crash when atomic builtins are called with pointer to zero-size struct (#GH90330)

- Clang now allows pointee types of atomic builtin arguments to be complete template types
  that was not instantiated elsewhere.

- ``__noop`` can now be used in a constant expression. (#GH102064)

- Fix ``__has_builtin`` incorrectly returning ``false`` for some C++ type traits. (#GH111477)

- Fix ``__builtin_source_location`` incorrectly returning wrong column for method chains. (#GH119129)

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a crash when an expression with a dependent ``__typeof__`` type is used as the operand of a unary operator. (#GH97646)
- Fixed incorrect pack expansion of init-capture references in requires expresssions.
- Fixed a failed assertion when checking invalid delete operator declaration. (#GH96191)
- Fix a crash when checking destructor reference with an invalid initializer. (#GH97230)
- Clang now correctly parses potentially declarative nested-name-specifiers in pointer-to-member declarators.
- Fix a crash when checking the initializer of an object that was initialized
  with a string literal. (#GH82167)
- Fix a crash when matching template template parameters with templates which have
  parameters of different class type. (#GH101394)
- Clang now correctly recognizes the correct context for parameter
  substitutions in concepts, so it doesn't incorrectly complain of missing
  module imports in those situations. (#GH60336)
- Fix init-capture packs having a size of one before being instantiated. (#GH63677)
- Clang now preserves the unexpanded flag in a lambda transform used for pack expansion. (#GH56852), (#GH85667),
  (#GH99877).
- Fixed a bug when diagnosing ambiguous explicit specializations of constrained member functions.
- Fixed an assertion failure when selecting a function from an overload set that includes a
  specialization of a conversion function template.
- Correctly diagnose attempts to use a concept name in its own definition;
  A concept name is introduced to its scope sooner to match the C++ standard. (#GH55875)
- Properly reject defaulted relational operators with invalid types for explicit object parameters,
  e.g., ``bool operator==(this int, const Foo&)`` (#GH100329), and rvalue reference parameters.
- Properly reject defaulted copy/move assignment operators that have a non-reference explicit object parameter.
- Clang now properly handles the order of attributes in `extern` blocks. (#GH101990).
- Fixed an assertion failure by preventing null explicit object arguments from being deduced. (#GH102025).
- Correctly check constraints of explicit instantiations of member functions. (#GH46029)
- When performing partial ordering of function templates, clang now checks that
  the deduction was consistent. Fixes (#GH18291).
- Fixed an assertion failure about a constraint of a friend function template references to a value with greater
  template depth than the friend function template. (#GH98258)
- Clang now rebuilds the template parameters of out-of-line declarations and specializations in the context
  of the current instantiation in all cases.
- Fix evaluation of the index of dependent pack indexing expressions/types specifiers (#GH105900)
- Correctly handle subexpressions of an immediate invocation in the presence of implicit casts. (#GH105558)
- Clang now correctly handles direct-list-initialization of a structured bindings from an array. (#GH31813)
- Mangle placeholders for deduced types as a template-prefix, such that mangling
  of template template parameters uses the correct production. (#GH106182)
- Fixed an assertion failure when converting vectors to int/float with invalid expressions. (#GH105486)
- Template parameter names are considered in the name lookup of out-of-line class template
  specialization right before its declaration context. (#GH64082)
- Fixed a constraint comparison bug for friend declarations. (#GH78101)
- Fix handling of ``_`` as the name of a lambda's init capture variable. (#GH107024)
- Fix an issue with dependent source location expressions (#GH106428), (#GH81155), (#GH80210), (#GH85373)
- Fixed a bug in the substitution of empty pack indexing types. (#GH105903)
- Clang no longer tries to capture non-odr used default arguments of template parameters of generic lambdas (#GH107048)
- Fixed a bug where defaulted comparison operators would remove ``const`` from base classes. (#GH102588)
- Fix a crash when using ``source_location`` in the trailing return type of a lambda expression. (#GH67134)
- A follow-up fix was added for (#GH61460), as the previous fix was not entirely correct. (#GH86361), (#GH112352)
- Fixed a crash in the typo correction of an invalid CTAD guide. (#GH107887)
- Fixed a crash when clang tries to substitute parameter pack while retaining the parameter
  pack. (#GH63819), (#GH107560)
- Fix a crash when a static assert declaration has an invalid close location. (#GH108687)
- Avoided a redundant friend declaration instantiation under a certain ``consteval`` context. (#GH107175)
- Fixed an assertion failure in debug mode, and potential crashes in release mode, when
  diagnosing a failed cast caused indirectly by a failed implicit conversion to the type of the constructor parameter.
- Fixed an assertion failure by adjusting integral to boolean vector conversions (#GH108326)
- Fixed a crash when mixture of designated and non-designated initializers in union. (#GH113855)
- Fixed an issue deducing non-type template arguments of reference type. (#GH73460)
- Fixed an issue in constraint evaluation, where type constraints on the lambda expression
  containing outer unexpanded parameters were not correctly expanded. (#GH101754)
- Fixes crashes with function template member specializations, and increases
  conformance of explicit instantiation behaviour with MSVC. (#GH111266)
- Fixed a bug in constraint expression comparison where the ``sizeof...`` expression was not handled properly
  in certain friend declarations. (#GH93099)
- Clang now instantiates the correct lambda call operator when a lambda's class type is
  merged across modules. (#GH110401)
- Fix a crash when parsing a pseudo destructor involving an invalid type. (#GH111460)
- Fixed an assertion failure when invoking recovery call expressions with explicit attributes
  and undeclared templates. (#GH107047), (#GH49093)
- Clang no longer crashes when a lambda contains an invalid block declaration that contains an unexpanded
  parameter pack. (#GH109148)
- Fixed overload handling for object parameters with top-level cv-qualifiers in explicit member functions (#GH100394)
- Fixed a bug in lambda captures where ``constexpr`` class-type objects were not properly considered ODR-used in
  certain situations. (#GH47400), (#GH90896)
- Fix erroneous templated array size calculation leading to crashes in generated code. (#GH41441)
- During the lookup for a base class name, non-type names are ignored. (#GH16855)
- Fix a crash when recovering an invalid expression involving an explicit object member conversion operator. (#GH112559)
- Clang incorrectly considered a class with an anonymous union member to not be
  const-default-constructible even if a union member has a default member initializer.
  (#GH95854).
- Fixed an assertion failure when evaluating an invalid expression in an array initializer. (#GH112140)
- Fixed an assertion failure in range calculations for conditional throw expressions. (#GH111854)
- Clang now correctly ignores previous partial specializations of member templates explicitly specialized for
  an implicitly instantiated class template specialization. (#GH51051)
- Fixed an assertion failure caused by invalid enum forward declarations. (#GH112208)
- Name independent data members were not correctly initialized from default member initializers. (#GH114069)
- Fixed expression transformation for ``[[assume(...)]]``, allowing using pack indexing expressions within the
  assumption if they also occur inside of a dependent lambda. (#GH114787)
- Lambdas now capture function types without considering top-level const qualifiers. (#GH84961)
- Clang now uses valid deduced type locations when diagnosing functions with trailing return type
  missing placeholder return type. (#GH78694)
- Fixed a bug where bounds of partially expanded pack indexing expressions were checked too early. (#GH116105)
- Fixed an assertion failure caused by using ``consteval`` in condition in consumed analyses. (#GH117385)
- Fixed an assertion failure caused by invalid default argument substitutions in non-defining
  friend declarations. (#GH113324)
- Fix a crash caused by incorrect argument position in merging deduced template arguments. (#GH113659)
- Fixed a parser crash when using pack indexing as a nested name specifier. (#GH119072) 
- Fixed a null pointer dereference issue when heuristically computing ``sizeof...(pack)`` expressions. (#GH81436)
- Fixed an assertion failure caused by mangled names with invalid identifiers. (#GH112205)
- Fixed an incorrect lambda scope of generic lambdas that caused Clang to crash when computing potential lambda
  captures at the end of a full expression. (#GH115931)
- Clang no longer rejects deleting a pointer of incomplete enumeration type. (#GH99278)
- Fixed recognition of ``std::initializer_list`` when it's surrounded with ``extern "C++"`` and exported
  out of a module (which is the case e.g. in MSVC's implementation of ``std`` module). (#GH118218)
- Fixed a pack expansion issue in checking unexpanded parameter sizes. (#GH17042)
- Fixed a bug where captured structured bindings were modifiable inside non-mutable lambda (#GH95081)
- Passing incomplete types to ``__is_base_of`` and other builtin type traits for which the corresponding
  standard type trait mandates a complete type is now a hard (non-sfinae-friendly) error
  (`LWG3929 <https://wg21.link/LWG3929>`__.) (#GH121278)
- Clang now identifies unexpanded parameter packs within the type constraint on a non-type template parameter. (#GH88866)
- Fixed an issue while resolving type of expression indexing into a pack of values of non-dependent type (#GH121242)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a crash that occurred when dividing by zero in complex integer division. (#GH55390).
- Fixed a bug in ``ASTContext::getRawCommentForAnyRedecl()`` where the function could
  sometimes incorrectly return null even if a comment was present. (#GH108145)
- Clang now correctly parses the argument of the ``relates``, ``related``, ``relatesalso``,
  and ``relatedalso`` comment commands.
- Clang now uses the location of the begin of the member expression for ``CallExpr``
  involving deduced ``this``. (#GH116928)
- Fixed printout of AST that uses pack indexing expression. (#GH116486)

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a crash in C due to incorrect lookup that members in nested anonymous struct/union
  can be found as ordinary identifiers in struct/union definition. (#GH31295)

- Fixed a crash caused by long chains of ``sizeof`` and other similar operators
  that can be followed by a non-parenthesized expression. (#GH45061)

- Fixed an crash when compiling ``#pragma STDC FP_CONTRACT DEFAULT`` with
  ``-ffp-contract=fast-honor-pragmas``. (#GH104830)

- Fixed a crash when function has more than 65536 parameters.
  Now a diagnostic is emitted. (#GH35741)

- Fixed ``-ast-dump`` crashes on codes involving ``concept`` with ``-ast-dump-decl-types``. (#GH94928)

- Fixed internal assertion firing when a declaration in the implicit global
  module is found through ADL. (GH#109879)

- Fixed a crash when an unscoped enumeration declared by an opaque-enum-declaration within a class template
  with a dependent underlying type is subject to integral promotion. (#GH117960)

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

- Clang now implements the Solaris-specific mangling of ``std::tm`` as
  ``tm``, same for ``std::div_t``, ``std::ldiv_t``, and
  ``std::lconv``, for Solaris ABI compatibility. (#GH33114)

AMDGPU Support
^^^^^^^^^^^^^^

- Initial support for gfx950

- Added headers ``gpuintrin.h`` and ``amdgpuintrin.h`` that contains common
  definitions for GPU builtin functions. This header can be included for OpenMP,
  CUDA, HIP, OpenCL, and C/C++.

NVPTX Support
^^^^^^^^^^^^^^

- Added headers ``gpuintrin.h`` and ``nvptxintrin.h`` that contains common
  definitions for GPU builtin functions. This header can be included for OpenMP,
  CUDA, HIP, OpenCL, and C/C++.

X86 Support
^^^^^^^^^^^

- The MMX vector intrinsic functions from ``*mmintrin.h`` which
  operate on `__m64` vectors, such as ``_mm_add_pi8``, have been
  reimplemented to use the SSE2 instruction-set and XMM registers
  unconditionally. These intrinsics are therefore *no longer
  supported* if MMX is enabled without SSE2 -- either from targeting
  CPUs from the Pentium-MMX through the Pentium 3, or explicitly via
  passing arguments such as ``-mmmx -mno-sse2``. MMX assembly code
  remains supported without requiring SSE2, including inside
  inline-assembly.

- The compiler builtins such as ``__builtin_ia32_paddb`` which
  formerly implemented the above MMX intrinsic functions have been
  removed. Any uses of these removed functions should migrate to the
  functions defined by the ``*mmintrin.h`` headers. A mapping can be
  found in the file ``clang/www/builtins.py``.

- Support ISA of ``AVX10.2``.
  * Supported MINMAX intrinsics of ``*_(mask(z)))_minmax(ne)_p[s|d|h|bh]`` and
  ``*_(mask(z)))_minmax_s[s|d|h]``.

- Supported intrinsics for ``SM4 and AVX10.2``.
  * Supported SM4 intrinsics of ``_mm512_sm4key4_epi32`` and
  ``_mm512_sm4rnds4_epi32``.

- All intrinsics in adcintrin.h can now be used in constant expressions.

- All intrinsics in adxintrin.h can now be used in constant expressions.

- All intrinsics in lzcntintrin.h can now be used in constant expressions.

- All intrinsics in bmiintrin.h can now be used in constant expressions.

- All intrinsics in bmi2intrin.h can now be used in constant expressions.

- All intrinsics in tbmintrin.h can now be used in constant expressions.

- Supported intrinsics for ``MOVRS AND AVX10.2``.
  * Supported intrinsics of ``_mm(256|512)_(mask(z))_loadrs_epi(8|16|32|64)``.
- Support ISA of ``AMX-FP8``.
- Support ISA of ``AMX-TRANSPOSE``.
- Support ISA of ``AMX-MOVRS``.
- Support ISA of ``AMX-AVX512``.
- Support ISA of ``AMX-TF32``.
- Support ISA of ``MOVRS``.

- Supported ``-march/tune=diamondrapids``

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

- In the ARM Target, the frame pointer (FP) of a leaf function can be retained
  by using the ``-fno-omit-frame-pointer`` option. If you want to eliminate the FP
  in leaf functions after enabling ``-fno-omit-frame-pointer``, you can do so by adding
  the ``-momit-leaf-frame-pointer`` option.

- Support has been added for the following processors (-mcpu identifiers in parenthesis):

  For AArch64:

  * FUJITSU-MONAKA (fujitsu-monaka)

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

- clang-cl now supports ``/std:c++23preview`` which enables C++23 features.

- Clang no longer allows references inside a union when emulating MSVC 1900+ even if `fms-extensions` is enabled.
  Starting with VS2015, MSVC 1900, this Microsoft extension is no longer allowed and always results in an error.
  Clang now follows the MSVC behavior in this scenario.
  When `-fms-compatibility-version=18.00` or prior is set on the command line this Microsoft extension is still
  allowed as VS2013 and prior allow it.

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^

- The option ``-mcmodel=large`` for the large code model is supported.
- Bump RVV intrinsic to version 1.0, the spec: https://github.com/riscv-non-isa/rvv-intrinsic-doc/releases/tag/v1.0.0-rc4

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed a bug about overriding a constexpr pure-virtual member function with a non-constexpr virtual member function which causes compilation failure when including standard C++ header `format`.

CUDA Support
^^^^^^^^^^^^
- Clang now supports CUDA SDK up to 12.6
- Added support for sm_100
- Added support for `__grid_constant__` attribute.
- CUDA now uses the new offloading driver by default. The new driver supports
  device-side LTO, interoperability with OpenMP and other languages, and native ``-fgpu-rdc``
  support with static libraries. The old behavior can be returned using the
  ``--no-offload-new-driver`` flag. The binary format is no longer compatible
  with the NVIDIA compiler's RDC-mode support. More information can be found at:
  https://clang.llvm.org/docs/OffloadingDesign.html

AIX Support
^^^^^^^^^^^

NetBSD Support
^^^^^^^^^^^^^^

WebAssembly Support
^^^^^^^^^^^^^^^^^^^

The default target CPU, "generic", now enables the `-mnontrapping-fptoint`
and `-mbulk-memory` flags, which correspond to the [Bulk Memory Operations]
and [Non-trapping float-to-int Conversions] language features, which are
[widely implemented in engines].

A new Lime1 target CPU is added, -mcpu=lime1. This CPU follows the definition of
the Lime1 CPU [here], and enables -mmultivalue, -mmutable-globals,
-mcall-indirect-overlong, -msign-ext, -mbulk-memory-opt, -mnontrapping-fptoint,
and -mextended-const.

[Bulk Memory Operations]: https://github.com/WebAssembly/bulk-memory-operations/blob/master/proposals/bulk-memory-operations/Overview.md
[Non-trapping float-to-int Conversions]: https://github.com/WebAssembly/spec/blob/master/proposals/nontrapping-float-to-int-conversion/Overview.md
[widely implemented in engines]: https://webassembly.org/features/
[here]: https://github.com/WebAssembly/tool-conventions/blob/main/Lime.md#lime1

AVR Support
^^^^^^^^^^^

- Reject C/C++ compilation for avr1 devices which have no SRAM.

DWARF Support in Clang
----------------------

Floating Point Support in Clang
-------------------------------

- Add ``__builtin_elementwise_atan2`` builtin for floating point types only.

Fixed Point Support in Clang
----------------------------

AST Matchers
------------

- Fixed an issue with the `hasName` and `hasAnyName` matcher when matching
  inline namespaces with an enclosing namespace of the same name.

- Fixed an ordering issue with the `hasOperands` matcher occurring when setting a
  binding in the first matcher and using it in the second matcher.

- Fixed a crash when traverse lambda expr with invalid captures. (#GH106444)

- Fixed ``isInstantiated`` and ``isInTemplateInstantiation`` to also match for variable templates. (#GH110666)

- Ensure ``hasName`` matches template specializations across inline namespaces,
  making `matchesNodeFullSlow` and `matchesNodeFullFast` consistent.

- Improved the performance of the ``getExpansionLocOfMacro`` by tracking already processed macros during recursion.

- Add ``exportDecl`` matcher to match export declaration.

- Ensure ``hasType`` and ``hasDeclaration`` match Objective-C interface declarations.

- Ensure ``pointee`` matches Objective-C pointer types.

- Add ``dependentScopeDeclRefExpr`` matcher to match expressions that refer to dependent scope declarations.

- Add ``dependentNameType`` matcher to match a dependent name type.

- Add ``dependentTemplateSpecializationType`` matcher to match a dependent template specialization type.

- Add ``hasDependentName`` matcher to match the dependent name of a DependentScopeDeclRefExpr or DependentNameType.

clang-format
------------

- Adds ``BreakBinaryOperations`` option.
- Adds ``TemplateNames`` option.
- Adds ``AlignFunctionDeclarations`` option to ``AlignConsecutiveDeclarations``.
- Adds ``IndentOnly`` suboption to ``ReflowComments`` to fix the indentation of
  multi-line comments without touching their contents, renames ``false`` to
  ``Never``, and ``true`` to ``Always``.
- Adds ``RemoveEmptyLinesInUnwrappedLines`` option.
- Adds ``KeepFormFeed`` option and set it to ``true`` for ``GNU`` style.
- Adds ``AllowShortNamespacesOnASingleLine`` option.
- Adds ``VariableTemplates`` option.
- Adds support for bash globstar in ``.clang-format-ignore``.
- Adds ``WrapNamespaceBodyWithEmptyLines`` option.

libclang
--------
- Add ``clang_isBeforeInTranslationUnit``. Given two source locations, it determines
  whether the first one comes strictly before the second in the source code.
- Add ``clang_getTypePrettyPrinted``.  It allows controlling the PrintingPolicy used
  to pretty-print a type.

Static Analyzer
---------------

New features
^^^^^^^^^^^^

- Now CSA models `__builtin_*_overflow` functions. (#GH102602)

- MallocChecker now checks for ``ownership_returns(class, idx)`` and ``ownership_takes(class, idx)``
  attributes with class names different from "malloc". Clang static analyzer now reports an error
  if class of allocation and deallocation function mismatches.
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#unix-mismatcheddeallocator-c-c>`__.

- Function effects, e.g. the ``nonblocking`` and ``nonallocating`` "performance constraint"
  attributes, are now verified. For example, for functions declared with the ``nonblocking``
  attribute, the compiler can generate warnings about the use of any language features, or calls to
  other functions, which may block.

- Introduced ``-warning-suppression-mappings`` flag to control diagnostic
  suppressions per file. See `documentation <https://clang.llvm.org/docs/WarningSuppressionMappings.html>_` for details.

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^

- In loops where the loop condition is opaque (i.e. the analyzer cannot
  determine whether it's true or false), the analyzer will no longer assume
  execution paths that perform more that two iterations. These unjustified
  assumptions caused false positive reports (e.g. 100+ out-of-bounds reports in
  the FFMPEG codebase) in loops where the programmer intended only two or three
  steps but the analyzer wasn't able to understand that the loop is limited.

Improvements
^^^^^^^^^^^^

- Improved the handling of the ``ownership_returns`` attribute. Now, Clang reports an
  error if the attribute is attached to a function that returns a non-pointer value.
  Fixes (#GH99501)

Moved checkers
^^^^^^^^^^^^^^

- The checker ``alpha.core.IdenticalExpr`` was deleted because it was
  duplicated in the clang-tidy checkers ``misc-redundant-expression`` and
  ``bugprone-branch-clone``.

- The checker ``alpha.security.MallocOverflow`` was deleted because it was
  badly implemented and its aggressive logic produced too many false positives.
  To detect too large arguments passed to malloc, consider using the checker
  ``alpha.taint.TaintedAlloc``.

- The checkers ``alpha.nondeterministic.PointerSorting`` and
  ``alpha.nondeterministic.PointerIteration`` were moved to a new bugprone
  checker named ``bugprone-nondeterministic-pointer-iteration-order``. The
  original checkers were implemented only using AST matching and make more
  sense as a single clang-tidy check.

- The checker ``alpha.unix.Chroot`` was modernized, improved and moved to
  ``unix.Chroot``. Testing was done on open source projects that use chroot(),
  and false issues addressed in the improvements based on real use cases. Open
  source projects used for testing include nsjail, lxroot, dive and ruri.
  This checker conforms to SEI Cert C recommendation `POS05-C. Limit access to
  files by creating a jail
  <https://wiki.sei.cmu.edu/confluence/display/c/POS05-C.+Limit+access+to+files+by+creating+a+jail>`_.
  Fixes (#GH34697).
  (#GH117791) [Documentation](https://clang.llvm.org/docs/analyzer/checkers.html#unix-chroot-c).

.. _release-notes-sanitizers:

Sanitizers
----------
- Introduced Realtime Sanitizer, activated by using the -fsanitize=realtime
  flag. This sanitizer detects unsafe system library calls, such as memory
  allocations and mutex locks. If any such function is called during invocation
  of a function marked with the ``[[clang::nonblocking]]`` attribute, an error
  is printed to the console and the process exits non-zero.

- Added the ``-fsanitize-undefined-ignore-overflow-pattern`` flag which can be
  used to disable specific overflow-dependent code patterns. The supported
  patterns are: ``add-signed-overflow-test``, ``add-unsigned-overflow-test``,
  ``negated-unsigned-const``, and ``unsigned-post-decr-while``. The sanitizer
  instrumentation can be toggled off for all available patterns by specifying
  ``all``. Conversely, you may disable all exclusions with ``none`` which is
  the default.

  .. code-block:: c++

     /// specified with ``-fsanitize-undefined-ignore-overflow-pattern=add-unsigned-overflow-test``
     int common_overflow_check_pattern(unsigned base, unsigned offset) {
       if (base + offset < base) { /* ... */ } // The pattern of `a + b < a`, and other re-orderings, won't be instrumented
     }

     /// specified with ``-fsanitize-undefined-ignore-overflow-pattern=add-signed-overflow-test``
     int common_overflow_check_pattern_signed(signed int base, signed int offset) {
       if (base + offset < base) { /* ... */ } // The pattern of `a + b < a`, and other re-orderings, won't be instrumented
     }

     /// specified with ``-fsanitize-undefined-ignore-overflow-pattern=negated-unsigned-const``
     void negation_overflow() {
       unsigned long foo = -1UL; // No longer causes a negation overflow warning
       unsigned long bar = -2UL; // and so on...
     }

     /// specified with ``-fsanitize-undefined-ignore-overflow-pattern=unsigned-post-decr-while``
     void while_post_decrement() {
       unsigned char count = 16;
       while (count--) { /* ... */ } // No longer causes unsigned-integer-overflow sanitizer to trip
     }

  Many existing projects have a large amount of these code patterns present.
  This new flag should allow those projects to enable integer sanitizers with
  less noise.

- ``-fsanitize=signed-integer-overflow``, ``-fsanitize=unsigned-integer-overflow``,
  ``-fsanitize=implicit-signed-integer-truncation``, ``-fsanitize=implicit-unsigned-integer-truncation``,
  ``-fsanitize=enum`` now properly support the
  "type" prefix within `Sanitizer Special Case Lists (SSCL)
  <https://clang.llvm.org/docs/SanitizerSpecialCaseList.html>`_. See that link
  for examples.

- Introduced an experimental Type Sanitizer, activated by using the
  ``-fsanitize=type`` flag. This sanitizer detects violations of C/C++ type-based
  aliasing rules.

- Implemented ``-f[no-]sanitize-trap=local-bounds``, and ``-f[no-]sanitize-recover=local-bounds``.

- ``-fsanitize-merge`` (default) and ``-fno-sanitize-merge`` have been added for
  fine-grained, unified control of which UBSan checks can potentially be merged
  by the compiler (for example,
  ``-fno-sanitize-merge=bool,enum,array-bounds,local-bounds``).

- Changed ``-fsanitize=pointer-overflow`` to no longer report ``NULL + 0`` as
  undefined behavior in C, in line with
  `N3322 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3322.pdf>`_,
  and matching the previous behavior for C++.
  ``NULL + non_zero`` continues to be reported as undefined behavior.

Python Binding Changes
----------------------
- Fixed an issue that led to crashes when calling ``Type.get_exception_specification_kind``.
- Added ``Cursor.pretty_printed``, a binding for ``clang_getCursorPrettyPrinted``,
  and related functions, which allow changing the formatting of pretty-printed code.
- Added ``Cursor.is_anonymous_record_decl``, a binding for
  ``clang_Cursor_isAnonymousRecordDecl``, which allows checking if a
  declaration is an anonymous union or anonymous struct.
- Added ``Type.pretty_printed`, a binding for ``clang_getTypePrettyPrinted``,
  which allows changing the formatting of pretty-printed types.

OpenMP Support
--------------
- Added support for 'omp assume' directive.
- Added support for 'omp scope' directive.
- Added support for allocator-modifier in 'allocate' clause.
- Changed the OpenMP DeviceRTL to use 'generic' IR. The
  ``LIBOMPTARGET_DEVICE_ARCHITECTURES`` CMake argument is now unused and will
  always build support for AMDGPU and NVPTX targets.
- Added support for combined masked constructs  'omp parallel masked taskloop',
  'omp parallel masked taskloop simd','omp masked taskloop' and 'omp masked taskloop simd' directive.

Improvements
^^^^^^^^^^^^
- Improve the handling of mapping array-section for struct containing nested structs with user defined mappers

- `num_teams` and `thead_limit` now accept multiple expressions when it is used
  along in ``target teams ompx_bare`` construct. This allows the target region
  to be launched with multi-dim grid on GPUs.

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
