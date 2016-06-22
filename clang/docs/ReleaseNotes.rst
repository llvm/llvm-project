=====================================
Clang 3.9 (In-Progress) Release Notes
=====================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 3.9 release. You may
   prefer the `Clang 3.8 Release Notes
   <http://llvm.org/releases/3.8.0/tools/clang/docs/ReleaseNotes.html>`_.

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

Note that if you are reading this file from a Subversion checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <http://llvm.org/releases/>`_.

What's New in Clang 3.9?
========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Clang will no longer passes --build-id by default to the linker. In modern
  linkers that is a relatively expensive option. It can be passed explicitly
  with -Wl,--build-id. To have clang always pass it, build clang with
  -DENABLE_LINKER_BUILD_ID.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang's diagnostics are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate source information
about them. The improvements since the 3.7 release include:

-  ...

New Compiler Flags
------------------

The option ....


New Pragmas in Clang
-----------------------

Clang now supports the ...

Windows Support
---------------

Clang's support for building native Windows programs ...

TLS is enabled for Cygwin defaults to -femulated-tls.


C Language Changes in Clang
---------------------------
The -faltivec and -maltivec flags no longer silently include altivec.h on Power platforms.

`RenderScript
<https://developer.android.com/guide/topics/renderscript/compute.html>`_
support added to the Frontend and enabled by the '-x renderscript' option or
the '.rs' file extension.

...

C11 Feature Support
^^^^^^^^^^^^^^^^^^^

...

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

...

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

...

Objective-C Language Changes in Clang
-------------------------------------

...

OpenCL C Language Changes in Clang
----------------------------------

...

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

Internal API Changes
--------------------

These are major API changes that have happened since the 3.8 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

-  ...

AST Matchers
------------

- has and hasAnyArgument: Matchers no longer ignores parentheses and implicit
  casts on the argument before applying the inner matcher. The fix was done to
  allow for greater control by the user. In all existing checkers that use this
  matcher all instances of code ``hasAnyArgument(<inner matcher>)`` or
  ``has(<inner matcher>)`` must be changed to
  ``hasAnyArgument(ignoringParenImpCasts(<inner matcher>))`` or
  ``has(ignoringParenImpCasts(<inner matcher>))``.

...

libclang
--------

...

Static Analyzer
---------------

...

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
