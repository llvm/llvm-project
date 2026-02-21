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

- Records carrying the trivial_abi attribute are now returned directly in registers
  in more cases when using the Microsoft ABI. It is not possible to pass trivial_abi
  records between MSVC and Clang, so there is no ABI compatibility requirement. This
  is an ABI break with old versions of Clang. (#GH87993)

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

- A new generic bit-reverse builtin function ``__builtin_bitreverseg`` that
  extends bit-reversal support to all standard integers type, including
  ``_BitInt``

New Compiler Flags
------------------
- New option ``-fms-anonymous-structs`` / ``-fno-ms-anonymous-structs`` added
  to enable or disable Microsoft's anonymous struct/union extension without
  enabling other ``-fms-extensions`` features (#GH177607).
- New option ``--precompile-reduced-bmi`` allows build system to generate a
  reduced BMI only for a C++20 importable module unit. Previously the users
  can only generate the reduced BMI as a by-product, e.g, an object files or
  a full BMI.

- New ``-cc1`` option ``-fexperimental-overflow-behavior-types`` added to
  enable parsing of the experimental ``overflow_behavior`` type attribute and
  type specifiers.

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------
- The `-mno-outline` and `-moutline` compiler flags are now allowed on RISC-V and X86, which both support the machine outliner.
- The `-mno-outline` flag will now add the `nooutline` IR attribute, so that
  `-mno-outline` and `-moutline` objects can be mixed correctly during LTO.

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

- Introduced a new type attribute ``__attribute__((overflow_behavior))`` which
  currently accepts either ``wrap`` or ``trap`` as an argument, enabling
  type-level control over overflow behavior. There is also an accompanying type
  specifier for each behavior kind via `__ob_wrap` and `__ob_trap`.

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

- Clang now emits ``-Wsizeof-pointer-memaccess`` when snprintf/vsnprintf use the sizeof 
  the destination buffer(dynamically allocated) in the len parameter(#GH162366)

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
- Fixed an assertion failure caused by error recovery while extending a nested name specifier with results from ordinary lookup. (#GH181470)

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed a behavioral discrepancy between deleted functions and private members when checking the ``enable_if`` attribute. (#GH175895)
- Fixed ``init_priority`` attribute by delaying type checks until after the type is deduced.

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed a crash when instantiating ``requires`` expressions involving substitution failures in C++ concepts. (#GH176402)
- Fixed a crash when a default argument is passed to an explicit object parameter. (#GH176639)
- Fixed a crash when diagnosing an invalid static member function with an explicit object parameter (#GH177741)
- Fixed a bug where captured variables in non-mutable lambdas were incorrectly treated as mutable 
  when used inside decltype in the return type. (#GH180460)
- Fixed a crash when evaluating uninitialized GCC vector/ext_vector_type vectors in ``constexpr``. (#GH180044)

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
- Fixed a crash when initializing a ``constexpr`` pointer with a floating-point literal in C23. (#GH180313)
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
- Add missing support for ``TraversalKind`` in some ``addMatcher()`` overloads.

clang-format
------------
- Add ``ObjCSpaceAfterMethodDeclarationPrefix`` option to control space between the
  '-'/'+' and the return type in Objective-C method declarations
- Add ``AfterComma`` value to ``BreakConstructorInitializers`` to allow breaking
  constructor initializers after commas, keeping the colon on the same line.

libclang
--------
- Fix crash in clang_getBinaryOperatorKindSpelling and clang_getUnaryOperatorKindSpelling

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
