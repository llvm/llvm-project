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

C/C++ Language Potentially Breaking Changes
-------------------------------------------

C++ Specific Potentially Breaking Changes
-----------------------------------------

- Clang now more aggressively optimizes away stores to objects after they are
  dead. This behavior can be disabled with ``-fno-lifetime-dse``.

ABI Changes in This Version
---------------------------

AST Dumping Potentially Breaking Changes
----------------------------------------

- The JSON AST dump now includes all fields from ``AvailabilityAttr``: ``platform``,
  ``introduced``, ``deprecated``, ``obsoleted``, ``unavailable``, ``message``,
  ``strict``, ``replacement``, ``priority``, and ``environment``. Previously, these
  fields were missing from the JSON output.

Clang Frontend Potentially Breaking Changes
-------------------------------------------

- HIPSPV toolchain: `--offload-targets=spirv{32,64}` option is
  deprecated and will be removed when the new offload driver becomes
  default. The replacement for the option is
  `--offload-targets=spirv{32,64}-unknown-chipstar` when using the new
  offload driver (`--offload-new-driver`).


Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------
- Remove ``CompletionString.Availability``. No libclang interfaces returned instances of it.
- ``CompletionString.availability`` now returns instances of ``CompletionString.AvailabilityKindCompat``.

  Instances of ``AvailabilityKindCompat`` have the same ``__str__`` representation
  as the previous ``CompletionChunk.Kind`` and are equality-comparable with
  the existing ``AvailabilityKind`` enum. It will be replaced by ``AvailabilityKind``
  in a future release. When this happens, the return type of ``CompletionString.availability``
  will change to ``AvailabilityKind``, so it is recommended to use ``AvailabilityKind``
  to compare with the return values of ``CompletionString.availability``.
- Remove ``availabilityKinds``. In this release, uses of ``availabilityKinds``
  need to be replaced by ``CompletionString.AvailabilityKind``.
- ``CompletionChunk.kind`` now returns instances of ``CompletionChunkKind``.

  Instances of ``CompletionChunkKind`` have the same ``__str__`` representation
  as the previous ``CompletionChunk.Kind`` for compatibility.
  These representations will be changed in a future release to match other enums.
- Remove ``completionChunkKindMap``. In this release, uses of ``completionChunkKindMap``
  need to be replaced by ``CompletionChunkKind``.
- Move ``SPELLING_CACHE`` into ``CompletionChunk`` and change it to use
  ``CompletionChunkKind`` instances as keys, instead of the enum values.
  An alias is kept in the form of a ``SPELLING_CACHE`` variable, but it only supports
  ``__getitem__`` and ``__contains__``. It will be removed in a future release.
  Please migrate to using ``CompletionChunk.SPELLING_CACHE`` instead.
- ``SourceLocation`` and ``SourceRange`` now use ``NotImplemented`` to delegate
  equality checks (``__eq__``) to the other object they are compared with when
  they are of different classes. They previously returned ``False`` when compared
  with objects of other classes.

What's New in Clang |release|?
==============================

C++ Language Changes
--------------------

- ``__is_trivially_equality_comparable`` no longer returns false for all enum types. (#GH132672)

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Clang now supports `P1857R3 <https://wg21.link/p1857r3>`_ Modules Dependency Discovery. (#GH54047)

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

- Added ``__builtin_stdc_rotate_left`` and ``__builtin_stdc_rotate_right``
  for bit rotation of unsigned integers including ``_BitInt`` types. Rotation
  counts are normalized modulo the bit-width and support negative values.
  Usable in constant expressions. Implicit conversion is supported for
  class/struct types with conversion operators.

<<<<<<< HEAD
- A new generic bit-reverse builtin function ``__builtin_bitreverseg`` that
  extends bit-reversal support to all standard integers type, including
  ``_BitInt``
=======
- Added ``__builtin_elementwise_ldexp``.

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
 
- A new generic byte swap builtin function ``__builtin_bswapg`` that extends the existing 
  __builtin_bswap{16,32,64} function family to support all standard integer types.

- A builtin ``__builtin_infer_alloc_token(<args>, ...)`` is provided to allow
  compile-time querying of allocation token IDs, where the builtin arguments
  mirror those normally passed to an allocation function.

- Clang now rejects the invalid use of ``constexpr`` with ``auto`` and an explicit type in C. (#GH163090)
>>>>>>> 3f06fd997749 ([Clang] Instantiate constexpr function when they are needed.)

New Compiler Flags
------------------
- New option ``-fms-anonymous-structs`` / ``-fno-ms-anonymous-structs`` added
  to enable or disable Microsoft's anonymous struct/union extension without
  enabling other ``-fms-extensions`` features (#GH177607).
- New option ``--precompile-reduced-bmi`` allows build system to generate a
  reduced BMI only for a C++20 importable module unit. Previously the users
  can only generate the reduced BMI as a by-product, e.g, an object files or
  a full BMI.

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------
- The `-mno-outline` and `-moutline` compiler flags are now allowed on RISC-V and X86, which both support the machine outliner.

Removed Compiler Flags
----------------------

Attribute Changes in Clang
--------------------------

- Added new attribute ``stack_protector_ignore`` to opt specific local variables out of
  the analysis which determines if a function should get a stack protector.  A function
  will still generate a stack protector if other local variables or command line flags
  require it.

- Added a new attribute, ``[[clang::no_outline]]`` to suppress outlining from
  annotated functions. This uses the LLVM `nooutline` attribute.

Improvements to Clang's diagnostics
-----------------------------------
- Added ``-Wlifetime-safety`` to enable lifetime safety analysis,
  a CFG-based intra-procedural analysis that detects use-after-free and related
  temporal safety bugs. See the
  `RFC <https://discourse.llvm.org/t/rfc-intra-procedural-lifetime-analysis-in-clang/86291>`_
  for more details. By design, this warning is enabled in ``-Weverything``. To disable
  the analysis, use ``-Wno-lifetime-safety`` or ``-fno-lifetime-safety``.

- Added ``-Wlifetime-safety-suggestions`` to enable lifetime annotation suggestions.
  This provides suggestions for function parameters that
  should be marked ``[[clang::lifetimebound]]`` based on lifetime analysis. For
  example, for the following function:

  .. code-block:: c++

    int* p(int *in) { return in; }

  Clang will suggest:

  .. code-block:: c++

    warning: parameter in intra-TU function should be marked [[clang::lifetimebound]]
    int* p(int *in) { return in; }
           ^~~~~~~
                   [[clang::lifetimebound]]
    note: param returned here
    int* p(int *in) { return in; }
                             ^~

- Added ``-Wlifetime-safety-noescape`` to detect misuse of ``[[clang::noescape]]``
  annotation where the parameter escapes through return. For example:

  .. code-block:: c++

    int* p(int *in [[clang::noescape]]) { return in; }

  Clang will warn:

  .. code-block:: c++

    warning: parameter is marked [[clang::noescape]] but escapes
    int* p(int *in [[clang::noescape]]) { return in; }
           ^~~~~~~
    note: returned here
    int* p(int *in [[clang::noescape]]) { return in; }
                                                 ^~

- Added ``-Wlifetime-safety-dangling-field`` to detect dangling field references
  when stack memory escapes to class fields. This is part of ``-Wlifetime-safety``
  and detects cases where local variables or parameters are stored in fields but
  outlive their scope. For example:

  .. code-block:: c++

<<<<<<< HEAD
<<<<<<< HEAD
    struct DanglingView {
      std::string_view view;
      DanglingView(std::string s) : view(s) {}  // warning: address of stack memory escapes to a field
    };

- Improved ``-Wassign-enum`` performance by caching enum enumerator values. (#GH176454)

- Fixed a false negative in ``-Warray-bounds`` where the warning was suppressed
  when accessing a member function on a past-the-end array element.
  (#GH179128)

- Added a missing space to the FixIt for the ``implicit-int`` group of diagnostics and 
  made sure that only one such diagnostic and FixIt is emitted per declaration group. (#GH179354)

- The ``-Wloop-analysis`` warning has been extended to catch more cases of
  variable modification inside lambda expressions (#GH132038).
=======
- Fixed a crash when enabling ``-fdiagnostics-format=sarif`` and the output
  carries messages like 'In file included from ...' or 'In module ...'.
  Now the include/import locations are written into `sarif.run.result.relatedLocations`.

- Clang now generates a fix-it for C++20 designated initializers when the
  initializers do not match the declaration order in the structure.
>>>>>>> 3f06fd997749 ([Clang] Instantiate constexpr function when they are needed.)
=======
- Fixed a crash when enabling ``-fdiagnostics-format=sarif`` and the output 
  carries messages like 'In file included from ...' or 'In module ...'.
  Now the include/import locations are written into `sarif.run.result.relatedLocations`.

- Clang now generates a fix-it for C++20 designated initializers when the 
  initializers do not match the declaration order in the structure. 
>>>>>>> ea6211a89115 (address more feedback)

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

- [MC/DC] Nested expressions are handled as individual MC/DC expressions.
- "Single byte coverage" now supports branch coverage and can be used
  together with ``-fcoverage-mcdc``.

Bug Fixes in This Version
-------------------------

- Fixed atomic boolean compound assignment; the conversion back to atomic bool would be miscompiled. (#GH33210)

- Fixed a failed assertion in the preprocessor when ``__has_embed`` parameters are missing parentheses. (#GH175088)
- Fix lifetime extension of temporaries in for-range-initializers in templates. (#GH165182)
- Fixed a preprocessor crash in ``__has_cpp_attribute`` on incomplete scoped attributes. (#GH178098)
- Fixes an assertion failure when evaluating ``__underlying_type`` on enum redeclarations. (#GH177943)
- Fixed an assertion failure caused by nested macro expansion during header-name lexing (``__has_embed(__has_include)``). (#GH178635)
- Clang now outputs relative paths of embeds for dependency output. (#GH161950)
- Fixed an assertion failure when evaluating ``_Countof`` on invalid ``void``-typed operands. (#GH180893)
- Fixed a ``-Winvalid-noreturn`` false positive for unreachable ``try`` blocks following an unconditional ``throw``. (#GH174822)

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed a behavioral discrepancy between deleted functions and private members when checking the ``enable_if`` attribute. (#GH175895)

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^
<<<<<<< HEAD
- Fixed a crash when instantiating ``requires`` expressions involving substitution failures in C++ concepts. (#GH176402)
- Fixed a crash when a default argument is passed to an explicit object parameter. (#GH176639)
- Fixed a crash when diagnosing an invalid static member function with an explicit object parameter (#GH177741)
- Fixed a crash when evaluating uninitialized GCC vector/ext_vector_type vectors in ``constexpr``. (#GH180044)
=======
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
  "initializing multiple members of union" coincide (#GH149985).
- Fix a crash when using ``explicit(bool)`` in pre-C++11 language modes. (#GH152729)
- Fix the parsing of variadic member functions when the ellipis immediately follows a default argument.(#GH153445)
- Fix a crash when using an explicit object parameter in a non-member function with an invalid return type.(#GH173943)
- Fixed a bug that caused ``this`` captured by value in a lambda with a dependent explicit object parameter to not be
  instantiated properly. (#GH154054)
- Fixed a bug where our ``member-like constrained friend`` checking caused an incorrect analysis of lambda captures. (#GH156225)
- Fixed a crash when implicit conversions from initialize list to arrays of
  unknown bound during constant evaluation. (#GH151716)
- Instantiate constexpr functions as needed before they are evaluated. (#GH73232) (#GH35052) (#GH100897)
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
- Fixed a template depth issue when parsing lambdas inside a type constraint. (#GH162092)
- Fix the support of zero-length arrays in SFINAE context. (#GH170040)
- Diagnose unresolved overload sets in non-dependent compound requirements. (#GH51246) (#GH97753)
- Fix a crash when extracting unavailable member type from alias in template deduction. (#GH165560)
- Fix incorrect diagnostics for lambdas with init-captures inside braced initializers. (#GH163498)
- Fixed an issue where templates prevented nested anonymous records from checking the deletion of special members. (#GH167217)
- Fixed serialization of pack indexing types, where we failed to expand those packs from a PCH/module. (#GH172464)
- Fixed spurious diagnoses of certain nested lambda expressions. (#GH149121) (#GH156579)
- Fix the result of ``__is_pointer_interconvertible_base_of`` when arguments are qualified and passed via template parameters. (#GH135273)
- Fixed a crash when evaluating nested requirements in requires-expressions that reference invented parameters. (#GH166325)
- Fixed a crash when standard comparison categories (e.g. ``std::partial_ordering``) are defined with incorrect static member types. (#GH170015) (#GH56571)
- Fixed a crash when parsing the ``enable_if`` attribute on C function declarations with identifier-list parameters. (#GH173826)
- Fixed an assertion failure triggered by nested lambdas during capture handling. (#GH172814)
- Fixed an assertion failure in vector conversions involving instantiation-dependent template expressions. (#GH173347)
>>>>>>> 3f06fd997749 ([Clang] Instantiate constexpr function when they are needed.)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed a bug where explicit nullability property attributes were not stored in AST nodes in Objective-C. (#GH179703)

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^
- Fixed the arguments of the format attribute on ``__builtin_os_log_format``.  Previously, they were off by 1.

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a crash when attempting to jump over initialization of a variable with variably modified type. (#GH175540)
- Fixed a crash when using loop hint with a value dependent argument inside a
  generic lambda. (#GH172289)
- Fixed a crash in C++ overload resolution with ``_Atomic``-qualified argument types. (#GH170433)
- Fixed an assertion when diagnosing address-space qualified ``new``/``delete`` in language-defined address spaces such as OpenCL ``__local``. (#GH178319)
- Fixed an assertion failure in ObjC++ ARC when binding a rvalue reference to reference with different lifetimes (#GH178524)

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

- Initial support for gfx1310

NVPTX Support
^^^^^^^^^^^^^^

X86 Support
^^^^^^^^^^^
- ``march=znver6`` is now supported.

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

- DWARF fission is now compatible with linker relaxations, allowing `-gsplit-dwarf` and `-mrelax`
  to be used together when building for the LoongArch platform.

RISC-V Support
^^^^^^^^^^^^^^

- Tenstorrent Ascalon D8 was renamed to Ascalon X. Use `tt-ascalon-x` with `-mcpu` or `-mtune`.

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

- Fixed a crash when ``__funcref`` is applied to a non-function pointer type.
  (#GH118233)

AVR Support
^^^^^^^^^^^

SystemZ Support
^^^^^^^^^^^^^^^

- Add support for `#pragma export` for z/OS.  This is a pragma used to export functions and variables
  with external linkage from shared libraries.  It provides compatibility with the IBM XL C/C++
  compiler.

DWARF Support in Clang
----------------------

Floating Point Support in Clang
-------------------------------

Fixed Point Support in Clang
----------------------------

AST Matchers
------------
- Add ``functionTypeLoc`` matcher for matching ``FunctionTypeLoc``.

clang-format
------------
- Add ``ObjCSpaceAfterMethodDeclarationPrefix`` option to control space between the 
  '-'/'+' and the return type in Objective-C method declarations

libclang
--------

Code Completion
---------------

- Fixed a crash in code completion when using a C-Style cast with a parenthesized
  operand in Objective-C++ mode. (#GH180125)

Static Analyzer
---------------

.. comment:
  This is for the Static Analyzer.
  Using the caret `^^^` underlining for subsections:
    - Crash and bug fixes
    - New checkers and features
    - Improvements
    - Moved checkers

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------
- Add deprecation warnings to ``CompletionChunk.isKind...`` methods.
  These will be removed in a future release. Existing uses should be adapted
  to directly compare equality of the ``CompletionChunk`` kind with
  the corresponding ``CompletionChunkKind`` variant.

  Affected methods: ``isKindOptional``, ``isKindTypedText``, ``isKindPlaceHolder``,
  ``isKindInformative`` and ``isKindResultType``.
- Add a deprecation warning to ``CodeCompletionResults.results``.
  This property will become an implementation detail with changed behavior in a 
  future release and should not be used directly.. Existing uses of 
  ``CodeCompletionResults.results`` should be changed to directly use
  ``CodeCompletionResults``: it nows supports ``__len__`` and ``__getitem__``,
  so it can be used the same as ``CodeCompletionResults.results``.

OpenMP Support
--------------
- Added support for ``transparent`` clause in task and taskloop directives.
- Added support for ``use_device_ptr`` clause to accept an optional
  ``fallback`` modifier (``fb_nullify`` or ``fb_preserve``) with OpenMP >= 61.

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
