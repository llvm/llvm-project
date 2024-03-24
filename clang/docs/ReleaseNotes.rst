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

- Setting the deprecated CMake variable ``GCC_INSTALL_PREFIX`` (which sets the
  default ``--gcc-toolchain=``) now leads to a fatal error.

C/C++ Language Potentially Breaking Changes
-------------------------------------------

C++ Specific Potentially Breaking Changes
-----------------------------------------
- Clang now diagnoses function/variable templates that shadow their own template parameters, e.g. ``template<class T> void T();``.
  This error can be disabled via `-Wno-strict-primary-template-shadow` for compatibility with previous versions of clang.

ABI Changes in This Version
---------------------------
- Fixed Microsoft name mangling of implicitly defined variables used for thread
  safe static initialization of static local variables. This change resolves
  incompatibilities with code compiled by MSVC but might introduce
  incompatibilities with code compiled by earlier versions of Clang when an
  inline member function that contains a static local variable with a dynamic
  initializer is declared with ``__declspec(dllimport)``. (#GH83616).

AST Dumping Potentially Breaking Changes
----------------------------------------

Clang Frontend Potentially Breaking Changes
-------------------------------------------
- Removed support for constructing on-stack ``TemplateArgumentList``s; interfaces should instead
  use ``ArrayRef<TemplateArgument>`` to pass template arguments. Transitioning internal uses to
  ``ArrayRef<TemplateArgument>`` reduces AST memory usage by 0.4% when compiling clang, and is
  expected to show similar improvements on other workloads.

- The ``-Wgnu-binary-literal`` diagnostic group no longer controls any
  diagnostics. Binary literals are no longer a GNU extension, they're now a C23
  extension which is controlled via ``-pedantic`` or ``-Wc23-extensions``. Use
  of ``-Wno-gnu-binary-literal`` will no longer silence this pedantic warning,
  which may break existing uses with ``-Werror``.

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

- Clang won't perform ODR checks for decls in the global module fragment any
  more to ease the implementation and improve the user's using experience.
  This follows the MSVC's behavior. Users interested in testing the more strict
  behavior can use the flag '-Xclang -fno-skip-odr-check-in-gmf'.
  (#GH79240).

- Implemented the `__is_layout_compatible` intrinsic to support
  `P0466R5: Layout-compatibility and Pointer-interconvertibility Traits <https://wg21.link/P0466R5>`_.

- Clang now implements [module.import]p7 fully. Clang now will import module
  units transitively for the module units coming from the same module of the
  current module units.
  Fixes `#84002 <https://github.com/llvm/llvm-project/issues/84002>`_.

- Initial support for class template argument deduction (CTAD) for type alias
  templates (`P1814R0 <https://wg21.link/p1814r0>`_).
  (#GH54051).

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P2718R0: Lifetime extension in range-based for loops <https://wg21.link/P2718R0>`_. Also
  materialize temporary object which is a prvalue in discarded-value expression.
- Implemented `P1774R8: Portable assumptions <https://wg21.link/P1774R8>`_.

- Implemented `P2448R2: Relaxing some constexpr restrictions <https://wg21.link/P2448R2>`_.

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P2662R3 Pack Indexing <https://wg21.link/P2662R3>`_.


Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Substitute template parameter pack, when it is not explicitly specified
  in the template parameters, but is deduced from a previous argument.
  (`#78449: <https://github.com/llvm/llvm-project/issues/78449>`_).

- Type qualifications are now ignored when evaluating layout compatibility
  of two types.
  (`CWG1719: Layout compatibility and cv-qualification revisited <https://cplusplus.github.io/CWG/issues/1719.html>`_).

- Alignment of members is now respected when evaluating layout compatibility
  of structs.
  (`CWG2583: Common initial sequence should consider over-alignment <https://cplusplus.github.io/CWG/issues/2583.html>`_).

- ``[[no_unique_address]]`` is now respected when evaluating layout
  compatibility of two types.
  (`CWG2759: [[no_unique_address] and common initial sequence  <https://cplusplus.github.io/CWG/issues/2759.html>`_).

C Language Changes
------------------

C23 Feature Support
^^^^^^^^^^^^^^^^^^^
- No longer diagnose use of binary literals as an extension in C23 mode. Fixes
  #GH72017.

- Corrected parsing behavior for the ``alignas`` specifier/qualifier in C23. We
  previously handled it as an attribute as in C++, but there are parsing
  differences. The behavioral differences are:

  .. code-block:: c

     struct alignas(8) /* was accepted, now rejected */ S {
       char alignas(8) /* was rejected, now accepted */ C;
     };
     int i alignas(8) /* was accepted, now rejected */ ;

  Fixes (#GH81472).

- Clang now generates predefined macros of the form ``__TYPE_FMTB__`` and
  ``__TYPE_FMTb__`` (e.g., ``__UINT_FAST64_FMTB__``) in C23 mode for use with
  macros typically exposed from ``<inttypes.h>``, such as ``PRIb8``.
  (`#81896: <https://github.com/llvm/llvm-project/issues/81896>`_).

- Clang now supports `N3018 The constexpr specifier for object definitions`
  <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3018.htm>`_.

Non-comprehensive list of changes in this release
-------------------------------------------------

- Added ``__builtin_readsteadycounter`` for reading fixed frequency hardware
  counters.

- ``__builtin_addc``, ``__builtin_subc``, and the other sizes of those
  builtins are now constexpr and may be used in constant expressions.

- Added ``__builtin_popcountg`` as a type-generic alternative to
  ``__builtin_popcount{,l,ll}`` with support for any unsigned integer type. Like
  the previous builtins, this new builtin is constexpr and may be used in
  constant expressions.

- Lambda expressions are now accepted in C++03 mode as an extension.

New Compiler Flags
------------------

- ``-Wmissing-designated-field-initializers``, grouped under ``-Wmissing-field-initializers``.
  This diagnostic can be disabled to make ``-Wmissing-field-initializers`` behave
  like it did before Clang 18.x. Fixes (`#56628 <https://github.com/llvm/llvm-project/issues/68933>`_)

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------
- Added a new diagnostic flag ``-Wreturn-mismatch`` which is grouped under
  ``-Wreturn-type``, and moved some of the diagnostics previously controlled by
  ``-Wreturn-type`` under this new flag. Fixes #GH72116.

- Added ``-Wcast-function-type-mismatch`` under the ``-Wcast-function-type``
  warning group. Moved the diagnostic previously controlled by
  ``-Wcast-function-type`` to the new warning group and added
  ``-Wcast-function-type-mismatch`` to ``-Wextra``. #GH76872

  .. code-block:: c

     int x(long);
     typedef int (f2)(void*);
     typedef int (f3)();

     void func(void) {
       // Diagnoses under -Wcast-function-type, -Wcast-function-type-mismatch,
       // -Wcast-function-type-strict, -Wextra
       f2 *b = (f2 *)x;
       // Diagnoses under -Wcast-function-type, -Wcast-function-type-strict
       f3 *c = (f3 *)x;
     }


Removed Compiler Flags
-------------------------

- The ``-freroll-loops`` flag has been removed. It had no effect since Clang 13.
- ``-m[no-]unaligned-access`` is removed for RISC-V and LoongArch.
  ``-m[no-]strict-align``, also supported by GCC, should be used instead.
  (`#85350 <https://github.com/llvm/llvm-project/pull/85350>`_.)

Attribute Changes in Clang
--------------------------
- Introduced a new function attribute ``__attribute__((amdgpu_max_num_work_groups(x, y, z)))`` or
  ``[[clang::amdgpu_max_num_work_groups(x, y, z)]]`` for the AMDGPU target. This attribute can be
  attached to HIP or OpenCL kernel function definitions to provide an optimization hint. The parameters
  ``x``, ``y``, and ``z`` specify the maximum number of workgroups for the respective dimensions,
  and each must be a positive integer when provided. The parameter ``x`` is required, while ``y`` and
  ``z`` are optional with default value of 1.

- The ``swiftasynccc`` attribute is now considered to be a Clang extension
  rather than a language standard feature. Please use
  ``__has_extension(swiftasynccc)`` to check the availability of this attribute
  for the target platform instead of ``__has_feature(swiftasynccc)``. Also,
  added a new extension query ``__has_extension(swiftcc)`` corresponding to the
  ``__attribute__((swiftcc))`` attribute.

Improvements to Clang's diagnostics
-----------------------------------
- Clang now applies syntax highlighting to the code snippets it
  prints.

- Clang now diagnoses member template declarations with multiple declarators.

- Clang now diagnoses use of the ``template`` keyword after declarative nested
  name specifiers.

- The ``-Wshorten-64-to-32`` diagnostic is now grouped under ``-Wimplicit-int-conversion`` instead
   of ``-Wconversion``. Fixes #GH69444.

- Clang now uses thousand separators when printing large numbers in integer overflow diagnostics.
  Fixes #GH80939.

- Clang now diagnoses friend declarations with an ``enum`` elaborated-type-specifier in language modes after C++98.

- Added diagnostics for C11 keywords being incompatible with language standards
  before C11, under a new warning group: ``-Wpre-c11-compat``.

- Now diagnoses an enumeration constant whose value is larger than can be
  represented by ``unsigned long long``, which can happen with a large constant
  using the ``wb`` or ``uwb`` suffix. The maximal underlying type is currently
  ``unsigned long long``, but this behavior may change in the future when Clang
  implements
  `WG14 N3029 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3029.htm>`_.
  (#GH69352).

- Clang now diagnoses extraneous template parameter lists as a language extension.

- Clang now diagnoses declarative nested name specifiers that name alias templates.

- Clang now diagnoses lambda function expressions being implicitly cast to boolean values, under ``-Wpointer-bool-conversion``.
  Fixes #GH82512.

- Clang now provides improved warnings for the ``cleanup`` attribute to detect misuse scenarios,
  such as attempting to call ``free`` on an unallocated object. Fixes
  `#79443 <https://github.com/llvm/llvm-project/issues/79443>`_.

- Clang no longer warns when the ``bitand`` operator is used with boolean
  operands, distinguishing it from potential typographical errors or unintended
  bitwise operations. Fixes #GH77601.

- Clang now correctly diagnoses no arguments to a variadic macro parameter as a C23/C++20 extension.
  Fixes #GH84495.

Improvements to Clang's time-trace
----------------------------------

Bug Fixes in This Version
-------------------------
- Fixed missing warnings when comparing mismatched enumeration constants
  in C (`#29217 <https://github.com/llvm/llvm-project/issues/29217>`).

- Clang now accepts elaborated-type-specifiers that explicitly specialize
  a member class template for an implicit instantiation of a class template.

- Fixed missing warnings when doing bool-like conversions in C23 (#GH79435).
- Clang's ``-Wshadow`` no longer warns when an init-capture is named the same as
  a class field unless the lambda can capture this.
  Fixes (#GH71976)

- Clang now accepts qualified partial/explicit specializations of variable templates that
  are not nominable in the lookup context of the specialization.

- Clang now doesn't produce false-positive warning `-Wconstant-logical-operand`
  for logical operators in C23.
  Fixes (#GH64356).

- ``__is_trivially_relocatable`` no longer returns ``false`` for volatile-qualified types.
  Fixes (#GH77091).

- Clang no longer produces a false-positive `-Wunused-variable` warning
  for variables created through copy initialization having side-effects in C++17 and later.
  Fixes (#GH64356) (#GH79518).

- Fix value of predefined macro ``__FUNCTION__`` in MSVC compatibility mode.
  Fixes (#GH66114).

- Clang now emits errors for explicit specializations/instatiations of lambda call
  operator.
  Fixes (#GH83267).

- Clang now correctly generates overloads for bit-precise integer types for
  builtin operators in C++. Fixes #GH82998.

- When performing mixed arithmetic between ``_Complex`` floating-point types and integers,
  Clang now correctly promotes the integer to its corresponding real floating-point
  type only rather than to the complex type (e.g. ``_Complex float / int`` is now evaluated
  as ``_Complex float / float`` rather than ``_Complex float / _Complex float``), as mandated
  by the C standard. This significantly improves codegen of `*` and `/` especially.
  Fixes (`#31205 <https://github.com/llvm/llvm-project/issues/31205>`_).

- Fixes an assertion failure on invalid code when trying to define member
  functions in lambdas.

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Fix crash when calling the constructor of an invalid class.
  (#GH10518) (#GH67914) (#GH78388)
- Fix crash when using lifetimebound attribute in function with trailing return.
  (#GH73619)
- Addressed an issue where constraints involving injected class types are perceived
  distinct from its specialization types. (#GH56482)
- Fixed a bug where variables referenced by requires-clauses inside
  nested generic lambdas were not properly injected into the constraint scope. (#GH73418)
- Fixed a crash where substituting into a requires-expression that refers to function
  parameters during the equivalence determination of two constraint expressions.
  (#GH74447)
- Fixed deducing auto& from const int in template parameters of partial
  specializations. (#GH77189)
- Fix for crash when using a erroneous type in a return statement.
  (#GH63244) (#GH79745)
- Fixed an out-of-bounds error caused by building a recovery expression for ill-formed
  function calls while substituting into constraints. (#GH58548)
- Fix incorrect code generation caused by the object argument
  of ``static operator()`` and ``static operator[]`` calls not being evaluated. (#GH67976)
- Fix crash and diagnostic with const qualified member operator new.
  Fixes (#GH79748)
- Fixed a crash where substituting into a requires-expression that involves parameter packs
  during the equivalence determination of two constraint expressions. (#GH72557)
- Fix a crash when specializing an out-of-line member function with a default
  parameter where we did an incorrect specialization of the initialization of
  the default parameter. (#GH68490)
- Fix a crash when trying to call a varargs function that also has an explicit object parameter.
  Fixes (#GH80971)
- Reject explicit object parameters on `new` and `delete` operators. (#GH82249)
- Fix a crash when trying to call a varargs function that also has an explicit object parameter. (#GH80971)
- Fixed a bug where abbreviated function templates would append their invented template parameters to
  an empty template parameter lists.
- Fix parsing of abominable function types inside type traits.
  Fixes (`#77585 <https://github.com/llvm/llvm-project/issues/77585>`_)
- Clang now classifies aggregate initialization in C++17 and newer as constant
  or non-constant more accurately. Previously, only a subset of the initializer
  elements were considered, misclassifying some initializers as constant. Partially fixes
  #GH80510.
- Clang now ignores top-level cv-qualifiers on function parameters in template partial orderings. (#GH75404)
- No longer reject valid use of the ``_Alignas`` specifier when declaring a
  local variable, which is supported as a C11 extension in C++. Previously, it
  was only accepted at namespace scope but not at local function scope.
- Clang no longer tries to call consteval constructors at runtime when they appear in a member initializer. (#GH82154)
- Fix crash when using an immediate-escalated function at global scope. (#GH82258)
- Correctly immediate-escalate lambda conversion functions. (#GH82258)
- Fixed an issue where template parameters of a nested abbreviated generic lambda within
  a requires-clause lie at the same depth as those of the surrounding lambda. This,
  in turn, results in the wrong template argument substitution during constraint checking.
  (#GH78524)
- Clang no longer instantiates the exception specification of discarded candidate function
  templates when determining the primary template of an explicit specialization.
- Fixed a crash in Microsoft compatibility mode where unqualified dependent base class
  lookup searches the bases of an incomplete class.
- Fix a crash when an unresolved overload set is encountered on the RHS of a ``.*`` operator.
  (#GH53815)
- In ``__restrict``-qualified member functions, attach ``__restrict`` to the pointer type of
  ``this`` rather than the pointee type.
  Fixes (#GH82941), (#GH42411) and (#GH18121).
- Clang now properly reports supported C++11 attributes when using
  ``__has_cpp_attribute`` and parses attributes with arguments in C++03 (#GH82995)
- Clang now properly diagnoses missing 'default' template arguments on a variety
  of templates. Previously we were diagnosing on any non-function template
  instead of only on class, alias, and variable templates, as last updated by
  CWG2032. Fixes (#GH83461)
- Fixed an issue where an attribute on a declarator would cause the attribute to
  be destructed prematurely. This fixes a pair of Chromium that were brought to
  our attention by an attempt to fix in (#GH77703). Fixes (#GH83385).
- Fix evaluation of some immediate calls in default arguments.
  Fixes (#GH80630)
- Fixed an issue where the ``RequiresExprBody`` was involved in the lambda dependency
  calculation. (#GH56556), (#GH82849).
- Fix a bug where overload resolution falsely reported an ambiguity when it was comparing
  a member-function against a non member function or a member-function with an
  explicit object parameter against a member function with no explicit object parameter
  when one of the function had more specialized templates.
  Fixes (`#82509 <https://github.com/llvm/llvm-project/issues/82509>`_)
  and (`#74494 <https://github.com/llvm/llvm-project/issues/74494>`_)
- Allow access to a public template alias declaration that refers to friend's
  private nested type. (#GH25708).
- Fixed a crash in constant evaluation when trying to access a
  captured ``this`` pointer in a lambda with an explicit object parameter.
  Fixes (#GH80997)
- Fix an issue where missing set friend declaration in template class instantiation.
  Fixes (#GH84368).
- Fixed a crash while checking constraints of a trailing requires-expression of a lambda, that the
  expression references to an entity declared outside of the lambda. (#GH64808)
- Clang's __builtin_bit_cast will now produce a constant value for records with empty bases. See:
  (#GH82383)
- Fix a crash when instantiating a lambda that captures ``this`` outside of its context. Fixes (#GH85343).
- Fix an issue where a namespace alias could be defined using a qualified name (all name components
  following the first `::` were ignored).

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Clang now properly preserves ``FoundDecls`` within a ``ConceptReference``. (#GH82628)

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Do not attempt to dump the layout of dependent types or invalid declarations
  when ``-fdump-record-layouts-complete`` is passed.
  Fixes (`#83684 <https://github.com/llvm/llvm-project/issues/83684>`_).

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

X86 Support
^^^^^^^^^^^

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

- ARMv7+ targets now default to allowing unaligned access, except Armv6-M, and
  Armv8-M without the Main Extension. Baremetal targets should check that the
  new default will work with their system configurations, since it requires
  that SCTLR.A is 0, SCTLR.U is 1, and that the memory in question is
  configured as "normal" memory. This brings Clang in-line with the default
  settings for GCC and Arm Compiler. Aside from making Clang align with other
  compilers, changing the default brings major performance and code size
  improvements for most targets. We have not changed the default behavior for
  ARMv6, but may revisit that decision in the future. Users can restore the old
  behavior with -m[no-]unaligned-access.
- An alias identifier (rdma) has been added for targeting the AArch64
  Architecture Extension which uses Rounding Doubling Multiply Accumulate
  instructions (rdm). The identifier is available on the command line as
  a feature modifier for -march and -mcpu as well as via target attributes
  like ``target_version`` or ``target_clones``.
- Support has been added for the following processors (-mcpu identifiers in parenthesis):
    * Arm Cortex-A78AE (cortex-a78ae).
    * Arm Cortex-A520AE (cortex-a520ae).
    * Arm Cortex-A720AE (cortex-a720ae).

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

- Clang-cl now supports function targets with intrinsic headers. This allows
  for runtime feature detection of intrinsics. Previously under clang-cl
  ``immintrin.h`` and similar intrinsic headers would only include the intrinsics
  if building with that feature enabled at compile time, e.g. ``avxintrin.h``
  would only be included if AVX was enabled at compile time. This was done to work
  around include times from MSVC STL including ``intrin.h`` under clang-cl.
  Clang-cl now provides ``intrin0.h`` for MSVC STL and therefore all intrinsic
  features without requiring enablement at compile time.
  Fixes: (`#53520 <https://github.com/llvm/llvm-project/issues/53520>`_)

- Improved compile times with MSVC STL. MSVC provides ``intrin0.h`` which is a
  header that only includes intrinsics that are used by MSVC STL to avoid the
  use of ``intrin.h``. MSVC STL when compiled under clang uses ``intrin.h``
  instead. Clang-cl now provides ``intrin0.h`` for the same compiler throughput
  purposes as MSVC. Clang-cl also provides ``yvals_core.h`` to redefine
  ``_STL_INTRIN_HEADER`` to expand to ``intrin0.h`` instead of ``intrin.h``.
  This also means that if all intrinsic features are enabled at compile time
  including STL headers will no longer slow down compile times since ``intrin.h``
  is not included from MSVC STL.

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^

- ``__attribute__((rvv_vector_bits(N)))`` is now supported for RVV vbool*_t types.
- Profile names in ``-march`` option are now supported.

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

- PTX is no longer included by default when compiling for CUDA. Using
  ``--cuda-include-ptx=all`` will return the old behavior.

CUDA Support
^^^^^^^^^^^^

AIX Support
^^^^^^^^^^^

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

- Support fixed point precision macros according to ``7.18a.3`` of
  `ISO/IEC TR 18037:2008 <https://standards.iso.org/ittf/PubliclyAvailableStandards/c051126_ISO_IEC_TR_18037_2008.zip>`_.

AST Matchers
------------

- ``isInStdNamespace`` now supports Decl declared with ``extern "C++"``.
- Add ``isExplicitObjectMemberFunction``.
- Fixed ``forEachArgumentWithParam`` and ``forEachArgumentWithParamType`` to
  not skip the explicit object parameter for operator calls.

clang-format
------------

- ``AlwaysBreakTemplateDeclarations`` is deprecated and renamed to
  ``BreakTemplateDeclarations``.
- ``AlwaysBreakAfterReturnType`` is deprecated and renamed to
  ``BreakAfterReturnType``.

libclang
--------

Static Analyzer
---------------

- Fixed crashing on loops if the loop variable was declared in switch blocks
  but not under any case blocks if ``unroll-loops=true`` analyzer config is
  set. (#GH68819)
- Support C++23 static operator calls. (#GH84972)

New features
^^^^^^^^^^^^

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^

Improvements
^^^^^^^^^^^^

- Support importing C++20 modules in clang-repl.

- Added support for ``TypeLoc::dump()`` for easier debugging, and improved
  textual and JSON dumping for various ``TypeLoc``-related nodes.

Moved checkers
^^^^^^^^^^^^^^

.. _release-notes-sanitizers:

Sanitizers
----------

- ``-fsanitize=signed-integer-overflow`` now instruments signed arithmetic even
  when ``-fwrapv`` is enabled. Previously, only division checks were enabled.

  Users with ``-fwrapv`` as well as a sanitizer group like
  ``-fsanitize=undefined`` or ``-fsanitize=integer`` enabled may want to
  manually disable potentially noisy signed integer overflow checks with
  ``-fno-sanitize=signed-integer-overflow``

Python Binding Changes
----------------------

- Exposed `CXRewriter` API as `class Rewriter`.
- Add some missing kinds from Index.h (CursorKind: 149-156, 272-320, 420-437.
  TemplateArgumentKind: 5-9. TypeKind: 161-175 and 178).

OpenMP Support
--------------

- Added support for the `[[omp::assume]]` attribute.

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
