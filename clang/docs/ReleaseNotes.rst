=======================
Clang 3.7 Release Notes
=======================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_


Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 3.7. Here we
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

What's New in Clang 3.7?
========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Use of the ``__declspec`` language extension for declaration attributes now
  requires passing the -fms-extensions or -fborland compiler flag. This language
  extension is also enabled when compiling CUDA code, but its use should be
  viewed as an implementation detail that is subject to change.

- On Windows targets, some uses of the ``__try``, ``__except``, and
  ``__finally`` language constructs are supported in Clang 3.7. MSVC-compatible
  C++ exceptions are not yet supported, however.

- Clang 3.7 fully supports OpenMP 3.1 and reported to work on many platforms,
  including x86, x86-64 and Power. Also, pragma ``omp simd`` from OpenMP 4.0 is
  supported as well. See below for details.

- Clang 3.7 includes an implementation of :doc:`control flow integrity
  <ControlFlowIntegrity>`, a security hardening mechanism.


Improvements to Clang's diagnostics
-----------------------------------

Clang's diagnostics are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate source information
about them. The improvements since the 3.6 release include:

- -Wrange-loop-analysis analyzes the loop variable type and the container type
  to determine whether copies are made of the container elements.  If possible,
  suggest a const reference type to prevent copies, or a non-reference type
  to indicate a copy is made.

- -Wredundant-move warns when a parameter variable is moved on return and the
  return type is the same as the variable.  Returning the variable directly
  will already make a move, so the call is not needed.

- -Wpessimizing-move warns when a local variable is moved on return and the
  return type is the same as the variable.  Copy elision cannot take place with
  a move, but can take place if the variable is returned directly.

- -Wmove is a new warning group which has the previous two warnings,
  -Wredundant-move and -Wpessimizing-move, as well as previous warning
  -Wself-move.  In addition, this group is part of -Wmost and -Wall now.

- -Winfinite-recursion, a warning for functions that only call themselves,
  is now part of -Wmost and -Wall.

- -Wobjc-circular-container prevents creation of circular containers, 
  it covers ``NSMutableArray``, ``NSMutableSet``, ``NSMutableDictionary``,
  ``NSMutableOrderedSet`` and all their subclasses.

New Compiler Flags
------------------

The sized deallocation feature of C++14 is now controlled by the
``-fsized-deallocation`` flag. This feature relies on library support that
isn't yet widely deployed, so the user must supply an extra flag to get the
extra functionality.


Objective-C Language Changes in Clang
-------------------------------------

- ``objc_boxable`` attribute was added. Structs and unions marked with this attribute can be
  used with boxed expressions (``@(...)``) to create ``NSValue``.

Profile Guided Optimization
---------------------------

Clang now accepts GCC-compatible flags for profile guided optimization (PGO).
You can now use ``-fprofile-generate=<dir>``, ``-fprofile-use=<dir>``,
``-fno-profile-generate`` and ``-fno-profile-use``. These flags have the
same semantics as their GCC counterparts. However, the generated profile
is still LLVM-specific. PGO profiles generated with Clang cannot be used
by GCC and vice-versa.

Clang now emits function entry counts in profile-instrumented binaries.
This has improved the computation of weights and frequencies in
profile analysis.

OpenMP Support
--------------
OpenMP 3.1 is fully supported, but disabled by default. To enable it, please use
the ``-fopenmp=libomp`` command line option. Your feedback (positive or negative) on
using OpenMP-enabled clang would be much appreciated; please share it either on
`cfe-dev <http://lists.llvm.org/mailman/listinfo/cfe-dev>`_ or `openmp-dev
<http://lists.llvm.org/mailman/listinfo/openmp-dev>`_ mailing lists.

In addition to OpenMP 3.1, several important elements of the 4.0 version of the
standard are supported as well:

- ``omp simd``, ``omp for simd`` and ``omp parallel for simd`` pragmas
- atomic constructs
- ``proc_bind`` clause of ``omp parallel`` pragma
- ``depend`` clause of ``omp task`` pragma (except for array sections)
- ``omp cancel`` and ``omp cancellation point`` pragmas
- ``omp taskgroup`` pragma

Internal API Changes
--------------------

These are major API changes that have happened since the 3.6 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

-  Some of the ``PPCallbacks`` interface now deals in ``MacroDefinition``
   objects instead of ``MacroDirective`` objects. This allows preserving
   full information on macros imported from modules.

-  ``clang-c/Index.h`` no longer ``#include``\s ``clang-c/Documentation.h``.
   You now need to explicitly ``#include "clang-c/Documentation.h"`` if
   you use the libclang documentation API.

Static Analyzer
---------------

* The generated plists now contain the name of the check that generated it.

* Configuration options can now be passed to the checkers (not just the static
  analyzer core).

* New check for dereferencing object that the result of a zero-length
  allocation.

* Also check functions in precompiled headers.

* Properly handle alloca() in some checkers.

* Various improvements to the retain count checker.


clang-tidy
----------
Added new checks:

* google-global-names-in-headers: flag global namespace pollution in header
  files.

* misc-assert-side-effect: detects ``assert()`` conditions with side effects
  which can cause different behavior in debug / release builds.

* misc-assign-operator-signature: finds declarations of assign operators with
  the wrong return and/or argument types.

* misc-inaccurate-erase: warns when some elements of a container are not
  removed due to using the ``erase()`` algorithm incorrectly.

* misc-inefficient-algorithm: warns on inefficient use of STL algorithms on
  associative containers.

* misc-macro-parentheses: finds macros that can have unexpected behavior due
  to missing parentheses.

* misc-macro-repeated-side-effects: checks for repeated argument with side
  effects in macros.

* misc-noexcept-move-constructor: flags user-defined move constructors and
  assignment operators not marked with ``noexcept`` or marked with
  ``noexcept(expr)`` where ``expr`` evaluates to ``false`` (but is not a
  ``false`` literal itself).

* misc-static-assert: replaces ``assert()`` with ``static_assert()`` if the
  condition is evaluable at compile time.

* readability-container-size-empty: checks whether a call to the ``size()``
  method can be replaced with a call to ``empty()``.

* readability-else-after-return: flags conditional statements having the
  ``else`` branch, when the ``true`` branch has a ``return`` as the last statement.

* readability-redundant-string-cstr: finds unnecessary calls to
  ``std::string::c_str()``.

* readability-shrink-to-fit: replaces copy and swap tricks on shrinkable
  containers with the ``shrink_to_fit()`` method call.

* readability-simplify-boolean-expr: looks for boolean expressions involving
  boolean constants and simplifies them to use the appropriate boolean
  expression directly (``if (x == true) ... -> if (x)``, etc.)

SystemZ
-------

* Clang will now always default to the z10 processor when compiling
  without any ``-march=`` option. Previous releases used to automatically
  detect the current host CPU when compiling natively. If you wish to
  still have clang detect the current host CPU, you now need to use the
  ``-march=native`` option.

* Clang now provides the ``<s390intrin.h>`` header file.

* Clang now supports the transactional-execution facility and
  provides associated builtins and the ``<htmintrin.h>`` and
  ``<htmxlintrin.h>`` header files. Support is enabled by default
  on zEC12 and above, and can additionally be enabled or disabled
  via the ``-mhtm`` / ``-mno-htm`` command line options.

* Clang now supports the vector facility. This includes a
  change in the ABI to pass arguments and return values of
  vector types in vector registers, as well as a change in
  the default alignment of vector types. Support is enabled
  by default on z13 and above, and can additionally be enabled
  or disabled via the ``-mvx`` / ``-mno-vx`` command line options.

* Clang now supports the System z vector language extension,
  providing a "vector" keyword to define vector types, and a
  set of builtins defined in the ``<vecintrin.h>`` header file.
  This can be enabled via the ``-fzvector`` command line option.
  For compatibility with GCC, Clang also supports the
  ``-mzvector`` option as an alias.
 
* Several cases of ABI incompatibility with GCC have been fixed.


Last release which will run on Windows XP and Windows Vista
-----------------------------------------------------------

This is expected to the be the last major release of Clang that will support
running on Windows XP and Windows Vista.  For the next major release the
minimum Windows version requirement will be Windows 7.

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
