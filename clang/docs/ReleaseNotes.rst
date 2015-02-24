=======================
Clang 3.6 Release Notes
=======================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 3.6. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <http://llvm.org/releases/3.6.0/docs/ReleaseNotes.html>`_.
All LLVM releases may be downloaded from the `LLVM releases web
site <http://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please check out the main please see the `Clang Web
Site <http://clang.llvm.org>`_ or the `LLVM Web
Site <http://llvm.org>`_.

What's New in Clang 3.6?
========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- The __has_attribute built-in macro no longer queries for attributes across
  multiple attribute syntaxes (GNU, C++11, __declspec, etc). Instead, it only
  queries GNU-style attributes. With the addition of __has_cpp_attribute and
  __has_declspec_attribute, this allows for more precise coverage of attribute
  syntax querying.

- clang-format now supports formatting Java code.


Improvements to Clang's diagnostics
-----------------------------------

Clang's diagnostics are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate source information
about them. The improvements since the 3.5 release include:

- Smarter typo correction. Clang now tries a bit harder to give a usable
  suggestion in more cases, and can now successfully recover in more
  situations where the suggestion changes how an expression is parsed.


New Compiler Flags
------------------

The ``-fpic`` option now uses small pic on PowerPC.


The __EXCEPTIONS macro
----------------------
``__EXCEPTIONS`` is now defined when landing pads are emitted, not when
C++ exceptions are enabled. The two can be different in Objective-C files:
If C++ exceptions are disabled but Objective-C exceptions are enabled,
landing pads will be emitted. Clang 3.6 is switching the behavior of
``__EXCEPTIONS``. Clang 3.5 confusingly changed the behavior of
``has_feature(cxx_exceptions)``, which used to be set if landing pads were
emitted, but is now set if C++ exceptions are enabled. So there are 3 cases:

Clang before 3.5:
   ``__EXCEPTIONS`` is set if C++ exceptions are enabled, ``cxx_exceptions``
   enabled if C++ or ObjC exceptions are enabled

Clang 3.5:
   ``__EXCEPTIONS`` is set if C++ exceptions are enabled, ``cxx_exceptions``
   enabled if C++ exceptions are enabled

Clang 3.6:
   ``__EXCEPTIONS`` is set if C++ or ObjC exceptions are enabled,
   ``cxx_exceptions`` enabled if C++ exceptions are enabled

To reliably test if C++ exceptions are enabled, use
``__EXCEPTIONS && __has_feature(cxx_exceptions)``, else things won't work in
all versions of Clang in Objective-C++ files.


New Pragmas in Clang
-----------------------

Clang now supports the `#pragma unroll` and `#pragma nounroll` directives to
specify loop unrolling optimization hints.  Placed just prior to the desired
loop, `#pragma unroll` directs the loop unroller to attempt to fully unroll the
loop.  The pragma may also be specified with a positive integer parameter
indicating the desired unroll count: `#pragma unroll _value_`.  The unroll count
parameter can be optionally enclosed in parentheses. The directive `#pragma
nounroll` indicates that the loop should not be unrolled.  These unrolling hints
may also be expressed using the `#pragma clang loop` directive.  See the Clang
`language extensions
<http://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-loop-hint-optimizations>`_
for details.

Windows Support
---------------

- Many, many bug fixes.

- Clang can now self-host using the ``msvc`` environment on x86 and x64
  Windows. This means that Microsoft C++ ABI is more or less feature-complete,
  minus exception support.

- Added more MSVC compatibility hacks, such as allowing more lookup into
  dependent bases of class templates when there is a known template pattern.
  As a result, applications using Active Template Library (ATL) or Windows
  Runtime Library (WRL) headers should compile correctly.

- Added support for the Visual C++ ``__super`` keyword.

- Added support for MSVC's ``__vectorcall`` calling convention, which is used
  in the upcoming Visual Studio 2015 STL.

- Added basic support for DWARF debug information in COFF files.


C Language Changes in Clang
---------------------------

- The default language mode for C compilations with Clang has been changed from
  C99 with GNU extensions to C11 with GNU extensions. C11 is largely
  backwards-compatible with C99, but if you want to restore the former behavior
  you can do so with the `-std=gnu99` flag.

C11 Feature Support
^^^^^^^^^^^^^^^^^^^

- Clang now provides an implementation of the standard C11 header `<stdatomic.h>`.

C++ Language Changes in Clang
-----------------------------

- An `upcoming change to C++ <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3922.html>_`
  changes the semantics of certain deductions of `auto` from a braced initializer
  list. Following the intent of the C++ committee, this change will be applied to
  our C++11 and C++14 modes as well as our experimental C++17 mode. Clang 3.6
  does not yet implement this change, but to provide a transition period, it
  warns on constructs whose meaning will change. The fix in all cases is to
  add an `=` prior to the left brace.

- Clang now supports putting identical constructors and destructors in
  the C5/D5 comdat, reducing code duplication.

- Clang will put individual ``.init_array/.ctors`` sections in
  comdats, reducing code duplication and speeding up startup.


C++17 Feature Support
^^^^^^^^^^^^^^^^^^^^^

Clang has experimental support for some proposed C++1z (tentatively, C++17)
features. This support can be enabled using the `-std=c++1z` flag.

New in Clang 3.6 is support for:

- Fold expressions

- `u8` character literals

- Nested namespace definitions: `namespace A::B { ... }` as a shorthand for
  `namespace A { namespace B { ... } }`

- Attributes for namespaces and enumerators

- Constant evaluation for all non-type template arguments

Note that these features may be changed or removed in future Clang releases
without notice.

Support for `for (identifier : range)` as a synonym for
`for (auto &&identifier : range)` has been removed as it is no longer currently
considered for C++17.

For more details on C++ feature support, see
`the C++ status page <http://clang.llvm.org/cxx_status.html>`_.


OpenMP Language Changes in Clang
--------------------------------

Clang 3.6 contains codegen for many individual OpenMP pragmas, but combinations are not completed yet.
We plan to continue codegen code drop aiming for completion in 3.7. Please see this link for up-to-date
`status <https://github.com/clang-omp/clang/wiki/Status-of-supported-OpenMP-constructs>_`.
LLVM's OpenMP runtime library, originally developed by Intel, has been modified to work on ARM, PowerPC,
as well as X86. The Runtime Library's compatibility with GCC 4.9 is improved
- missed entry points added, barrier and fork/join code improved, one more type of barrier enabled.
Support for ppc64le architecture is now available and automatically detected when using cmake system.
Using makefile the new "ppc64le" arch type is available.
Contributors to this work include AMD, Argonne National Lab., IBM, Intel, Texas Instruments, University of Houston and many others.


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
list <http://lists.cs.uiuc.edu/mailman/listinfo/cfe-dev>`_.
