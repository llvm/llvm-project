=======================
Clang 3.9 Release Notes
=======================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 3.9. Here we
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

What's New in Clang 3.9?
========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Clang will no longer pass ``--build-id`` by default to the linker. In modern
  linkers that is a relatively expensive option. It can be passed explicitly
  with ``-Wl,--build-id``. To have clang always pass it, build clang with
  ``-DENABLE_LINKER_BUILD_ID``.
- On Itanium ABI targets, attribute abi_tag is now supported for compatibility
  with GCC. Clang's implementation of abi_tag is mostly compatible with GCC ABI
  version 10.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang's diagnostics are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate source information
about them. The improvements since the 3.8 release include:

- ``-Wcomma`` is a new warning to show most uses of the builtin comma operator.

- ``-Wfloat-conversion`` has two new sub-warnings to give finer grain control for
  floating point to integer conversion warnings.

  - ``-Wfloat-overflow-conversion`` detects when a constant floating point value
    is converted to an integer type and will overflow the target type.

  - ``-Wfloat-zero-conversion`` detects when a non-zero floating point value is
    converted to a zero integer value.

Attribute Changes in Clang
--------------------------

- The ``nodebug`` attribute may now be applied to static, global, and local
  variables (but not parameters or non-static data members). This will suppress
  all debugging information for the variable (and its type, if there are no
  other uses of the type).


Windows Support
---------------

TLS is enabled for Cygwin and defaults to -femulated-tls.

Proper support, including correct mangling and overloading, added for
MS-specific "__unaligned" type qualifier.

clang-cl now has limited support for the precompiled header flags /Yc, /Yu, and
/Fp.  If the precompiled header is passed on the compile command with /FI, then
the precompiled header flags are honored.  But if the precompiled header is
included by an ``#include <stdafx.h>`` in each source file instead of by a
``/FIstdafx.h`` flag, these flag continue to be ignored.

clang-cl has a new flag, ``/imsvc <dir>``, for adding a directory to the system
include search path (where warnings are disabled by default) without having to
set ``%INCLUDE``.

C Language Changes in Clang
---------------------------
The -faltivec and -maltivec flags no longer silently include altivec.h on Power platforms.

`RenderScript
<https://developer.android.com/guide/topics/renderscript/compute.html>`_
support has been added to the frontend and enabled by the '-x renderscript'
option or the '.rs' file extension.


C++ Language Changes in Clang
-----------------------------

- Clang now enforces the rule that a *using-declaration* cannot name an enumerator of a
  scoped enumeration.

  .. code-block:: c++

    namespace Foo { enum class E { e }; }
    namespace Bar {
      using Foo::E::e; // error
      constexpr auto e = Foo::E::e; // ok
    }

- Clang now enforces the rule that an enumerator of an unscoped enumeration declared at
  class scope can only be named by a *using-declaration* in a derived class.

  .. code-block:: c++

    class Foo { enum E { e }; }
    using Foo::e; // error
    static constexpr auto e = Foo::e; // ok


C++1z Feature Support
^^^^^^^^^^^^^^^^^^^^^

Clang's experimental support for the upcoming C++1z standard can be enabled with ``-std=c++1z``.
Changes to C++1z features since Clang 3.8:

- The ``[[fallthrough]]``, ``[[nodiscard]]``, and ``[[maybe_unused]]`` attributes are
  supported in C++11 onwards, and are largely synonymous with Clang's existing attributes
  ``[[clang::fallthrough]]``, ``[[gnu::warn_unused_result]]``, and ``[[gnu::unused]]``.
  Use ``-Wimplicit-fallthrough`` to warn on unannotated fallthrough within ``switch``
  statements.

- In C++1z mode, aggregate initialization can be performed for classes with base classes:

  .. code-block:: c++

    struct A { int n; };
    struct B : A { int x, y; };
    B b = { 1, 2, 3 }; // b.n == 1, b.x == 2, b.y == 3

- The range in a range-based ``for`` statement can have different types for its ``begin``
  and ``end`` iterators. This is permitted as an extension in C++11 onwards.

- Lambda-expressions can explicitly capture ``*this`` (to capture the surrounding object
  by copy). This is permitted as an extension in C++11 onwards.

- Objects of enumeration type can be direct-list-initialized from a value of the underlying
  type. ``E{n}`` is equivalent to ``E(n)``, except that it implies a check for a narrowing
  conversion.

- Unary *fold-expression*\s over an empty pack are now rejected for all operators
  other than ``&&``, ``||``, and ``,``.

OpenCL C Language Changes in Clang
----------------------------------

Clang now has support for all OpenCL 2.0 features.  In particular, the following
features have been completed since the previous release:

- Pipe builtin functions (s6.13.16.2-4).
- Address space conversion functions ``to_{global/local/private}``.
- ``nosvm`` attribute support.
- Improved diagnostic and generation of Clang Blocks used in OpenCL kernel code.
- ``opencl_unroll_hint`` pragma.

Several miscellaneous improvements have been made:

- Supported extensions are now part of the target representation to give correct
  diagnostics for unsupported target features during compilation. For example,
  when compiling for a target that does not support the double precision
  floating point extension, Clang will give an error when encountering the
  ``cl_khr_fp64`` pragma. Several missing extensions were added covering up to
  and including OpenCL 2.0.
- Clang now comes with the OpenCL standard headers declaring builtin types and
  functions up to and including OpenCL 2.0 in ``lib/Headers/opencl-c.h``. By
  default, Clang will not include this header. It can be included either using
  the regular ``-I<path to header location>`` directive or (if the default one
  from installation is to be used) using the ``-finclude-default-header``
  frontend flag.

  Example:

  .. code-block:: none

    echo "bool is_wg_uniform(int i){return get_enqueued_local_size(i)==get_local_size(i);}" > test.cl
    clang -cc1 -finclude-default-header -cl-std=CL2.0 test.cl

  All builtin function declarations from OpenCL 2.0 will be automatically
  visible in test.cl.
- Image types have been improved with better diagnostics for access qualifiers.
  Images with one access qualifier type cannot be used in declarations for
  another type. Also qualifiers are now propagated from the frontend down to
  libraries and backends.
- Diagnostic improvements for OpenCL types, address spaces and vectors.
- Half type literal support has been added. For example, ``1.0h`` represents a
  floating point literal in half precision, i.e., the value ``0xH3C00``.
- The Clang driver now accepts OpenCL compiler options ``-cl-*`` (following the
  OpenCL Spec v1.1-1.2 s5.8). For example, the ``-cl-std=CL1.2`` option from the
  spec enables compilation for OpenCL 1.2, or ``-cl-mad-enable`` will enable
  fusing multiply-and-add operations.
- Clang now uses function metadata instead of module metadata to propagate
  information related to OpenCL kernels e.g. kernel argument information.

OpenMP Support in Clang
----------------------------------

Added support for all non-offloading features from OpenMP 4.5, including using
data members in private clauses of non-static member functions. Additionally,
data members can be used as loop control variables in loop-based directives.

Currently Clang supports OpenMP 3.1 and all non-offloading features of
OpenMP 4.0/4.5. Offloading features are under development. Clang defines macro
_OPENMP and sets it to OpenMP 3.1 (in accordance with OpenMP standard) by
default. User may change this value using ``-fopenmp-version=[31|40|45]`` option.

The codegen for OpenMP constructs was significantly improved to produce much
more stable and faster code.

AST Matchers
------------

- has and hasAnyArgument: Matchers no longer ignore parentheses and implicit
  casts on the argument before applying the inner matcher. The fix was done to
  allow for greater control by the user. In all existing checkers that use this
  matcher all instances of code ``hasAnyArgument(<inner matcher>)`` or
  ``has(<inner matcher>)`` must be changed to
  ``hasAnyArgument(ignoringParenImpCasts(<inner matcher>))`` or
  ``has(ignoringParenImpCasts(<inner matcher>))``.

Static Analyzer
---------------

The analyzer now checks for incorrect usage of MPI APIs in C and C++. This
check can be enabled by passing the following command to scan-build:
``-enable-checker optin.mpi.MPI-Checker.``

The analyzer now checks for improper instance cleanup up in Objective-C
``-dealloc`` methods under manual retain/release.

On Windows, checks for memory leaks, double frees, and use-after-free problems
are now enabled by default.

The analyzer now includes scan-build-py, an experimental reimplementation of
scan-build in Python that also creates compilation databases.

The scan-build tool now supports a ``--force-analyze-debug-code`` flag that
forces projects to analyze in debug mode. This flag leaves in assertions and so
typically results in fewer false positives.


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
