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

Non-comprehensive list of changes in this release
-------------------------------------------------

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

Attribute Changes in Clang
--------------------------

Improvements to Clang's diagnostics
-----------------------------------
- Clang now applies syntax highlighting to the code snippets it
  prints.

- Clang now diagnoses member template declarations with multiple declarators.

Improvements to Clang's time-trace
----------------------------------

Bug Fixes in This Version
-------------------------
- Clang now accepts elaborated-type-specifiers that explicitly specialize
  a member class template for an implicit instantiation of a class template.

- Fixed missing warnings when doing bool-like conversions in C23 (`#79435 <https://github.com/llvm/llvm-project/issues/79435>`_).

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

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
- Fix incorrect code generation caused by the object argument of ``static operator()`` and ``static operator[]`` calls not being evaluated.
  Fixes (`#67976 <https://github.com/llvm/llvm-project/issues/67976>`_)

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

AST Matchers
------------

clang-format
------------

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
