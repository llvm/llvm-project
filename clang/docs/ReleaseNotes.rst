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

Target OS macros extension
^^^^^^^^^^^^^^^^^^^^^^^^^^
A new Clang extension (see :ref:`here <target_os_detail>`) is enabled for
Darwin (Apple platform) targets. Clang now defines ``TARGET_OS_*`` macros for
these targets, which could break existing code bases with improper checks for
the ``TARGET_OS_`` macros. For example, existing checks might fail to include
the ``TargetConditionals.h`` header from Apple SDKs and therefore leaving the
macros undefined and guarded code unexercised.

Affected code should be checked to see if it's still intended for the specific
target and fixed accordingly.

The extension can be turned off by the option ``-fno-define-target-os-macros``
as a workaround.

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
  (`#79240 <https://github.com/llvm/llvm-project/issues/79240>`_).

- Implemented the `__is_layout_compatible` intrinsic to support
  `P0466R5: Layout-compatibility and Pointer-interconvertibility Traits <https://wg21.link/P0466R5>`_.
  Note: `CWG1719: Layout compatibility and cv-qualification revisited  <https://cplusplus.github.io/CWG/issues/1719.html>`_
  and `CWG2759: [[no_unique_address] and common initial sequence <https://cplusplus.github.io/CWG/issues/2759.html>`_
  are not yet implemented.

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

C Language Changes
------------------

C23 Feature Support
^^^^^^^^^^^^^^^^^^^
- No longer diagnose use of binary literals as an extension in C23 mode. Fixes
  `#72017 <https://github.com/llvm/llvm-project/issues/72017>`_.

- Corrected parsing behavior for the ``alignas`` specifier/qualifier in C23. We
  previously handled it as an attribute as in C++, but there are parsing
  differences. The behavioral differences are:

  .. code-block:: c

     struct alignas(8) /* was accepted, now rejected */ S {
       char alignas(8) /* was rejected, now accepted */ C;
     };
     int i alignas(8) /* was accepted, now rejected */ ;

  Fixes (`#81472 <https://github.com/llvm/llvm-project/issues/81472>`_).

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

.. _target_os_detail:

Target OS macros extension
^^^^^^^^^^^^^^^^^^^^^^^^^^
A pair of new flags ``-fdefine-target-os-macros`` and
``-fno-define-target-os-macros`` has been added to Clang to enable/disable the
extension to provide built-in definitions of a list of ``TARGET_OS_*`` macros
based on the target triple.

The extension is enabled by default for Darwin (Apple platform) targets.

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
   of ``-Wconversion``. Fixes `#69444 <https://github.com/llvm/llvm-project/issues/69444>`_.

- Clang now diagnoses friend declarations with an ``enum`` elaborated-type-specifier in language modes after C++98.

- Added diagnostics for C11 keywords being incompatible with language standards
  before C11, under a new warning group: ``-Wpre-c11-compat``.

Improvements to Clang's time-trace
----------------------------------

Bug Fixes in This Version
-------------------------
- Clang now accepts elaborated-type-specifiers that explicitly specialize
  a member class template for an implicit instantiation of a class template.

- Fixed missing warnings when doing bool-like conversions in C23 (`#79435 <https://github.com/llvm/llvm-project/issues/79435>`_).
- Clang's ``-Wshadow`` no longer warns when an init-capture is named the same as
  a class field unless the lambda can capture this.
  Fixes (`#71976 <https://github.com/llvm/llvm-project/issues/71976>`_)

- Clang now accepts qualified partial/explicit specializations of variable templates that
  are not nominable in the lookup context of the specialization.

- Clang now doesn't produce false-positive warning `-Wconstant-logical-operand`
  for logical operators in C23.
  Fixes (`#64356 <https://github.com/llvm/llvm-project/issues/64356>`_).

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Fix crash when calling the constructor of an invalid class.
  Fixes (`#10518 <https://github.com/llvm/llvm-project/issues/10518>`_),
  (`#67914 <https://github.com/llvm/llvm-project/issues/10518>`_),
  and (`#78388 <https://github.com/llvm/llvm-project/issues/78388>`_)
- Fix crash when using lifetimebound attribute in function with trailing return.
  Fixes (`#73619 <https://github.com/llvm/llvm-project/issues/73619>`_)
- Addressed an issue where constraints involving injected class types are perceived
  distinct from its specialization types.
  (`#56482 <https://github.com/llvm/llvm-project/issues/56482>`_)
- Fixed a bug where variables referenced by requires-clauses inside
  nested generic lambdas were not properly injected into the constraint scope.
  (`#73418 <https://github.com/llvm/llvm-project/issues/73418>`_)
- Fixed a crash where substituting into a requires-expression that refers to function
  parameters during the equivalence determination of two constraint expressions.
  (`#74447 <https://github.com/llvm/llvm-project/issues/74447>`_)
- Fixed deducing auto& from const int in template parameters of partial
  specializations.
  (`#77189 <https://github.com/llvm/llvm-project/issues/77189>`_)
- Fix for crash when using a erroneous type in a return statement.
  Fixes (`#63244 <https://github.com/llvm/llvm-project/issues/63244>`_)
  and (`#79745 <https://github.com/llvm/llvm-project/issues/79745>`_)
- Fixed an out-of-bounds error caused by building a recovery expression for ill-formed
  function calls while substituting into constraints.
  (`#58548 <https://github.com/llvm/llvm-project/issues/58548>`_)
- Fix incorrect code generation caused by the object argument of ``static operator()`` and ``static operator[]`` calls not being evaluated.
  Fixes (`#67976 <https://github.com/llvm/llvm-project/issues/67976>`_)
- Fix crash and diagnostic with const qualified member operator new.
  Fixes (`#79748 <https://github.com/llvm/llvm-project/issues/79748>`_)
- Fixed a crash where substituting into a requires-expression that involves parameter packs
  during the equivalence determination of two constraint expressions.
  (`#72557 <https://github.com/llvm/llvm-project/issues/72557>`_)
- Fix a crash when specializing an out-of-line member function with a default
  parameter where we did an incorrect specialization of the initialization of
  the default parameter.
  Fixes (`#68490 <https://github.com/llvm/llvm-project/issues/68490>`_)
- Fix a crash when trying to call a varargs function that also has an explicit object parameter.
  Fixes (`#80971 ICE when explicit object parameter be a function parameter pack`)
- Fixed a bug where abbreviated function templates would append their invented template parameters to
  an empty template parameter lists.
- Clang now classifies aggregate initialization in C++17 and newer as constant
  or non-constant more accurately. Previously, only a subset of the initializer
  elements were considered, misclassifying some initializers as constant. Fixes
  some of (`#80510 <https://github.com/llvm/llvm-project/issues/80510>`).
- Clang now ignores top-level cv-qualifiers on function parameters in template partial orderings.
  (`#75404 <https://github.com/llvm/llvm-project/issues/75404>`_)
- No longer reject valid use of the ``_Alignas`` specifier when declaring a
  local variable, which is supported as a C11 extension in C++. Previously, it
  was only accepted at namespace scope but not at local function scope.
- Clang no longer tries to call consteval constructors at runtime when they appear in a member initializer.
  (`#782154 <https://github.com/llvm/llvm-project/issues/82154>`_`)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
