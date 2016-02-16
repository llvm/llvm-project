=======================
Clang 3.8 Release Notes
=======================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 3.8. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <http://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <http://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please check out the main please see the `Clang Web
Site <http://clang.llvm.org>`_ or the `LLVM Web
Site <http://llvm.org>`_.

What's New in Clang 3.8?
========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Feature1...

Improvements to Clang's diagnostics
-----------------------------------

Clang's diagnostics are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate source information
about them. The improvements since the 3.7 release include:

- ``-Wmicrosoft`` has been split into many targeted flags, so that projects can
  choose to enable only a subset of these warnings. ``-Wno-microsoft`` still
  disables all these warnings, and ``-Wmicrosoft`` still enables them all.

-  ...

New Compiler Flags
------------------

Clang can "tune" DWARF debugging information to suit one of several different
debuggers. This fine-tuning can mean omitting DWARF features that the
debugger does not need or use, or including DWARF extensions specific to the
debugger. Clang supports tuning for three debuggers, as follows.

- ``-ggdb`` is equivalent to ``-g`` plus tuning for the GDB debugger. For
  compatibility with GCC, Clang allows this option to be followed by a
  single digit from 0 to 3 indicating the debugging information "level."
  For example, ``-ggdb1`` is equivalent to ``-ggdb -g1``.

- ``-glldb`` is equivalent to ``-g`` plus tuning for the LLDB debugger.

- ``-gsce`` is equivalent to ``-g`` plus tuning for the Sony Computer
  Entertainment debugger.

Specifying ``-g`` without a tuning option will use a target-dependent default.


New Pragmas in Clang
-----------------------

Clang now supports the ...

Windows Support
---------------

Clang's support for building native Windows programs ...


C Language Changes in Clang
---------------------------

Better support for ``__builtin_object_size``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang 3.8 has expanded support for the ``__builtin_object_size`` intrinsic.
Specifically, ``__builtin_object_size`` will now fail less often when you're
trying to get the size of a subobject. Additionally, the ``pass_object_size``
attribute was added, which allows ``__builtin_object_size`` to successfully
report the size of function parameters, without requiring that the function be
inlined.


``overloadable`` attribute relaxations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously, functions marked ``overloadable`` in C would strictly use C++'s
type conversion rules, so the following code would not compile:

.. code-block:: c

  void foo(char *bar, char *baz) __attribute__((overloadable));
  void foo(char *bar) __attribute__((overloadable));

  void callFoo() {
    int a;
    foo(&a);
  }

Now, Clang is able to selectively use C's type conversion rules during overload
resolution in C, which allows the above example to compile (albeit potentially
with a warning about an implicit conversion from ``int*`` to ``char*``).


...


C11 Feature Support
^^^^^^^^^^^^^^^^^^^

...

C++ Language Changes in Clang
-----------------------------

- ...

C++11 Feature Support
^^^^^^^^^^^^^^^^^^^^^

...

Objective-C Language Changes in Clang
-------------------------------------

...

OpenCL C Language Changes in Clang
----------------------------------

Several OpenCL 2.0 features have been added, including:

- Command-line option ``-std=CL2.0``.

- Generic address space (``__generic``) along with new conversion rules
  between different address spaces and default address space deduction.

- Support for program scope variables with ``__global`` address space.

- Pipe specifier was added (although no pipe functions are supported yet).

- Atomic types: ``atomic_int``, ``atomic_uint``, ``atomic_long``,
  ``atomic_ulong``, ``atomic_float``, ``atomic_double``, ``atomic_flag``,
  ``atomic_intptr_t``, ``atomic_uintptr_t``, ``atomic_size_t``,
  ``atomic_ptrdiff_t`` and their usage with C11 style builtin functions.

- Image types: ``image2d_depth_t``, ``image2d_array_depth_t``,
  ``image2d_msaa_t``, ``image2d_array_msaa_t``, ``image2d_msaa_depth_t``,
  ``image2d_array_msaa_depth_t``.

- Other types (for pipes and device side enqueue): ``clk_event_t``,
  ``queue_t``, ``ndrange_t``, ``reserve_id_t``.

Several additional features/bugfixes have been added to the previous standards:

- A set of floating point arithmetic relaxation flags: ``-cl-no-signed-zeros``,
  ``-cl-unsafe-math-optimizations``, ``-cl-finite-math-only``,
  ``-cl-fast-relaxed-math``.

- Added ``^^`` to the list of reserved operations.

- Improved vector support and diagnostics.

- Improved diagnostics for function pointers.

CUDA Support in Clang
---------------------
Clang has experimental support for end-to-end CUDA compilation now:

- The driver now detects CUDA installation, creates host and device compilation
  pipelines, links device-side code with appropriate CUDA bitcode and produces
  single object file with host and GPU code.

- Implemented target attribute-based function overloading which allows clang to
  compile CUDA sources without splitting them into separate host/device TUs.

Internal API Changes
--------------------

These are major API changes that have happened since the 3.7 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

* With this release, the autoconf build system is deprecated. It will be removed
  in the 3.9 release. Please migrate to using CMake. For more information see:
  `Building LLVM with CMake <http://llvm.org/docs/CMake.html>`_

AST Matchers
------------
The AST matcher functions were renamed to reflect the exact AST node names,
which is a breaking change to AST matching code. The following matchers were
affected:

=======================	============================
Previous Matcher Name	New Matcher Name
=======================	============================
recordDecl		recordDecl and cxxRecordDecl
ctorInitializer		cxxCtorInitializer
constructorDecl		cxxConstructorDecl
destructorDecl		cxxDestructorDecl
methodDecl		cxxMethodDecl
conversionDecl		cxxConversionDecl
memberCallExpr		cxxMemberCallExpr
constructExpr		cxxConstructExpr
unresolvedConstructExpr	cxxUnresolvedConstructExpr
thisExpr		cxxThisExpr
bindTemporaryExpr	cxxBindTemporaryExpr
newExpr			cxxNewExpr
deleteExpr		cxxDeleteExpr
defaultArgExpr		cxxDefaultArgExpr
operatorCallExpr	cxxOperatorCallExpr
forRangeStmt		cxxForRangeStmt
catchStmt		cxxCatchStmt
tryStmt			cxxTryStmt
throwExpr		cxxThrowExpr
boolLiteral		cxxBoolLiteral
nullPtrLiteralExpr	cxxNullPtrLiteralExpr
reinterpretCastExpr	cxxReinterpretCastExpr
staticCastExpr		cxxStaticCastExpr
dynamicCastExpr		cxxDynamicCastExpr
constCastExpr		cxxConstCastExpr
functionalCastExpr	cxxFunctionalCastExpr
temporaryObjectExpr	cxxTemporaryObjectExpr
CUDAKernalCallExpr	cudaKernelCallExpr
=======================	============================

recordDecl() previously matched AST nodes of type CXXRecordDecl, but now
matches AST nodes of type RecordDecl. If a CXXRecordDecl is required, use the
cxxRecordDecl() matcher instead.

...

libclang
--------

...

Static Analyzer
---------------

The scan-build and scan-view tools will now be installed with clang. Use these
tools to run the static analyzer on projects and view the produced results.

Static analysis of C++ lambdas has been greatly improved, including
interprocedural analysis of lambda applications.

Several new checks were added:

- The analyzer now checks for misuse of ``vfork()``.
- The analyzer can now detect excessively-padded structs. This check can be
  enabled by passing the following command to scan-build:
  ``-enable-checker optin.performance.Padding``.
- The checks to detect misuse of ``_Nonnull`` type qualifiers as well as checks
  to detect misuse of Objective-C generics were added.
- The analyzer now has opt in checks to detect localization errors in Cocoa
  applications. The checks warn about uses of non-localized ``NSStrings``
  passed to UI methods expecting localized strings and on ``NSLocalizedString``
  macros that are missing the comment argument. These can be enabled by passing
  the following command to scan-build:
  ``-enable-checker optin.osx.cocoa.localizability``.

Core Analysis Improvements
==========================

- ...

New Issues Found
================

- ...

Python Binding Changes
----------------------

The following methods have been added:

-  ...

Significant Known Problems
==========================

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <http://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Subversion version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us via the `mailing
list <http://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
