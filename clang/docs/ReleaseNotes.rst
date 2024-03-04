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

C++ Specific Potentially Breaking Changes
-----------------------------------------
- Clang now diagnoses function/variable templates that shadow their own template parameters, e.g. ``template<class T> void T();``.
  This error can be disabled via `-Wno-strict-primary-template-shadow` for compatibility with previous versions of clang.

ABI Changes in This Version
---------------------------

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

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P2718R0: Lifetime extension in range-based for loops <https://wg21.link/P2718R0>`_. Also
  materialize temporary object which is a prvalue in discarded-value expression.

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

Non-comprehensive list of changes in this release
-------------------------------------------------

- Added ``__builtin_readsteadycounter`` for reading fixed frequency hardware
  counters.

- ``__builtin_addc``, ``__builtin_subc``, and the other sizes of those
  builtins are now constexpr and may be used in constant expressions.

New Compiler Flags
------------------

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
-------------------------

- The ``-freroll-loops`` flag has been removed. It had no effect since Clang 13.

Attribute Changes in Clang
--------------------------

Improvements to Clang's diagnostics
-----------------------------------
- Clang now applies syntax highlighting to the code snippets it
  prints.

- Clang now diagnoses member template declarations with multiple declarators.

- Clang now diagnoses use of the ``template`` keyword after declarative nested
  name specifiers.

- The ``-Wshorten-64-to-32`` diagnostic is now grouped under ``-Wimplicit-int-conversion`` instead
   of ``-Wconversion``. Fixes #GH69444.

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

- Clang no longer produces a false-positive `-Wunused-variable` warning
  for variables created through copy initialization having side-effects in C++17 and later.
  Fixes (#GH64356) (#GH79518).

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

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

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

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^

- ``__attribute__((rvv_vector_bits(N)))`` is now supported for RVV vbool*_t types.

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

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
