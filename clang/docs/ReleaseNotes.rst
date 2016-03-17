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
documentation <../../../docs/ReleaseNotes.html>`_. All LLVM
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

Improvements to Clang's diagnostics
-----------------------------------

Clang's diagnostics are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate source information
about them. The improvements since the 3.7 release include:

- ``-Wmicrosoft`` has been split into many targeted flags, so that projects can
  choose to enable only a subset of these warnings. ``-Wno-microsoft`` still
  disables all these warnings, and ``-Wmicrosoft`` still enables them all.

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

The new ``-fstrict-vtable-pointers`` flag enables better devirtualization
support (experimental).


Alignment
---------
Clang has gotten better at passing down strict type alignment information to LLVM,
and several targets have gotten better at taking advantage of that information.

Dereferencing a pointer that is not adequately aligned for its type is undefined
behavior.  It may crash on target architectures that strictly enforce alignment, but
even on architectures that do not, frequent use of unaligned pointers may hurt
the performance of the generated code.

If you find yourself fixing a bug involving an inadequately aligned pointer, you
have several options.

The best option, when practical, is to increase the alignment of the memory.
For example, this array is not guaranteed to be sufficiently aligned to store
a pointer value:

.. code-block:: c

  char buffer[sizeof(const char*)];

Writing a pointer directly into it violates C's alignment rules:

.. code-block:: c

  ((const char**) buffer)[0] = "Hello, world!\n";

But you can use alignment attributes to increase the required alignment:

.. code-block:: c

  __attribute__((aligned(__alignof__(const char*))))
  char buffer[sizeof(const char*)];

When that's not practical, you can instead reduce the alignment requirements
of the pointer.  If the pointer is to a struct that represents that layout of a
serialized structure, consider making that struct packed; this will remove any
implicit internal padding that the compiler might add to the struct and
reduce its alignment requirement to 1.

.. code-block:: c

  struct file_header {
    uint16_t magic_number;
    uint16_t format_version;
    uint16_t num_entries;
  } __attribute__((packed));

You may also override the default alignment assumptions of a pointer by
using a typedef with explicit alignment:

.. code-block:: c

  typedef const char *unaligned_char_ptr __attribute__((aligned(1)));
  ((unaligned_char_ptr*) buffer)[0] = "Hello, world!\n";

The final option is to copy the memory into something that is properly
aligned.  Be aware, however, that Clang will assume that pointers are
properly aligned for their type when you pass them to a library function
like memcpy.  For example, this code will assume that the source and
destination pointers are both properly aligned for an int:

.. code-block:: c

  void copy_int_array(int *dest, const int *src, size_t num) {
    memcpy(dest, src, num * sizeof(int));
  }

You may explicitly disable this assumption by casting the argument to a
less-aligned pointer type:

.. code-block:: c

  void copy_unaligned_int_array(int *dest, const int *src, size_t num) {
    memcpy((char*) dest, (const char*) src, num * sizeof(int));
  }

Clang promises not to look through the explicit cast when inferring the
alignment of this memcpy.


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

OpenMP Support in Clang
-----------------------

OpenMP 3.1 is fully supported and is enabled by default with ``-fopenmp`` 
which now uses the Clang OpenMP library instead of the GCC OpenMP library.
The runtime can be built in-tree.  

In addition to OpenMP 3.1, several important elements of the OpenMP 4.0/4.5 
are supported as well. We continue to aim to complete OpenMP 4.5

- ``map`` clause
- task dependencies
- ``num_teams`` clause
- ``thread_limit`` clause
- ``target`` and ``target data`` directive
- ``target`` directive with implicit data mapping
- ``target enter data`` and ``target exit data`` directive
- Array sections [2.4, Array Sections].
- Directive name modifiers for ``if`` clause [2.12, if Clause].
- ``linear`` clause can be used in loop-based directives [2.7.2, loop Construct].
- ``simdlen`` clause [2.8, SIMD Construct].
- ``hint`` clause [2.13.2, critical Construct].
- Parsing/semantic analysis of all non-device directives introduced in OpenMP 4.5.

The codegen for OpenMP constructs was significantly improved allowing us to produce much more stable and fast code.
Full test cases of IR are also implemented.

CUDA Support in Clang
---------------------
Clang has experimental support for end-to-end CUDA compilation now:

- The driver now detects CUDA installation, creates host and device compilation
  pipelines, links device-side code with appropriate CUDA bitcode and produces
  single object file with host and GPU code.

- Implemented target attribute-based function overloading which allows Clang to
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


Static Analyzer
---------------

The scan-build and scan-view tools will now be installed with Clang. Use these
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


Clang-tidy
----------

New checks have been added to clang-tidy:

* Checks enforcing certain rules of the `CERT Secure Coding Standards
  <https://www.securecoding.cert.org/confluence/display/seccode/SEI+CERT+Coding+Standards>`_:

  * `cert-dcl03-c <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-dcl03-c.html>`_
  * `cert-dcl50-cpp <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-dcl50-cpp.html>`_
  * `cert-err52-cpp <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-err52-cpp.html>`_
  * `cert-err58-cpp <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-err58-cpp.html>`_
  * `cert-err60-cpp <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-err60-cpp.html>`_
  * `cert-err61-cpp <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-err61-cpp.html>`_
  * `cert-fio38-c <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-fio38-c.html>`_
  * `cert-oop11-cpp <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-oop11-cpp.html>`_

* Checks supporting the `C++ Core Guidelines
  <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`_:

  * `cppcoreguidelines-pro-bounds-array-to-pointer-decay <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-bounds-array-to-pointer-decay.html>`_
  * `cppcoreguidelines-pro-bounds-constant-array-index <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-bounds-constant-array-index.html>`_
  * `cppcoreguidelines-pro-bounds-pointer-arithmetic <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-bounds-pointer-arithmetic.html>`_
  * `cppcoreguidelines-pro-type-const-cast <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-const-cast.html>`_
  * `cppcoreguidelines-pro-type-cstyle-cast <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-cstyle-cast.html>`_
  * `cppcoreguidelines-pro-type-reinterpret-cast <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-reinterpret-cast.html>`_
  * `cppcoreguidelines-pro-type-static-cast-downcast <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-static-cast-downcast.html>`_
  * `cppcoreguidelines-pro-type-union-access <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-union-access.html>`_
  * `cppcoreguidelines-pro-type-vararg <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-vararg.html>`_

* The functionality of the clang-modernize tool has been moved to the new
  ``modernize`` module in clang-tidy along with a few new checks:

  * `modernize-loop-convert <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-loop-convert.html>`_
  * `modernize-make-unique <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-make-unique.html>`_
  * `modernize-pass-by-value <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-pass-by-value.html>`_
  * `modernize-redundant-void-arg <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-redundant-void-arg.html>`_
  * `modernize-replace-auto-ptr <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-replace-auto-ptr.html>`_
  * `modernize-shrink-to-fit <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-shrink-to-fit.html>`_ (renamed from readability-shrink-to-fit)
  * `modernize-use-auto <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-use-auto.html>`_
  * `modernize-use-default <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-use-default.html>`_
  * `modernize-use-nullptr <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-use-nullptr.html>`_
  * `modernize-use-override <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-use-override.html>`_ (renamed from misc-use-override)

* New checks flagging various readability-related issues:

  * `readability-identifier-naming <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/readability-identifier-naming.html>`_
  * `readability-implicit-bool-cast <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/readability-implicit-bool-cast.html>`_
  * `readability-inconsistent-declaration-parameter-name <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/readability-inconsistent-declaration-parameter-name.html>`_
  * `readability-uniqueptr-delete-release <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/readability-uniqueptr-delete-release.html>`_

* New ``performance`` module for checks targeting potential performance issues:

  * performance-unnecessary-copy-initialization

* A few new checks have been added to the ``misc`` module:

  * `misc-definitions-in-headers <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-definitions-in-headers.html>`_
  * misc-move-const-arg
  * `misc-move-constructor-init <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-move-constructor-init.html>`_
  * `misc-new-delete-overloads <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-new-delete-overloads.html>`_
  * `misc-non-copyable-objects <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-non-copyable-objects.html>`_
  * `misc-sizeof-container <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-sizeof-container.html>`_
  * `misc-string-integer-assignment <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-string-integer-assignment.html>`_
  * `misc-throw-by-value-catch-by-reference <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-throw-by-value-catch-by-reference.html>`_
  * `misc-unused-alias-decls <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-unused-alias-decls.html>`_
  * `misc-unused-parameters <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-unused-parameters.html>`_
  * `misc-virtual-near-miss <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-virtual-near-miss.html>`_


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
