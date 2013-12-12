=====================================
Clang 3.4 (In-Progress) Release Notes
=====================================

.. contents::
   :local:
   :depth: 2

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 3.4. Here we
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

What's New in Clang 3.4?
========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Last release which will build as C++98
--------------------------------------

This is expected to be the last release of Clang which compiles using a C++98
toolchain. We expect to start using some C++11 features in Clang starting after
this release. That said, we are committed to supporting a reasonable set of
modern C++ toolchains as the host compiler on all of the platforms. This will
at least include Visual Studio 2012 on Windows, and Clang 3.1 or GCC 4.7.x on
Mac and Linux. The final set of compilers (and the C++11 features they support)
is not set in stone, but we wanted users of Clang to have a heads up that the
next release will involve a substantial change in the host toolchain
requirements.

Note that this change is part of a change for the entire LLVM project, not just
Clang.

Major New Features
------------------

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang's diagnostics are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate source information
about them. The improvements since the 3.3 release include:

- -Wheader-guard warns on mismatches between the #ifndef and #define lines
  in a header guard.
- -Wlogical-not-parentheses warns when a logical not ('!') only applies to the
  left-hand side of a comparison.  This warning is part of -Wparentheses.
- Boolean increment, a deprecated feature, has own warning flag
  -Wdeprecated-increment-bool, and is still part of -Wdeprecated.
- Clang errors on builtin enum increments and decrements.
- -Wloop-analysis now warns on for-loops which have the same increment or 
  decrement in the loop header as the last statement in the loop.
- -Wuninitialized now performs checking across field initializers to detect
  when one field in used uninitialized in another field initialization.
- Clang can detect initializer list use inside a macro and suggest parentheses
  if possible to fix.
- Many improvements to Clang's typo correction facilities, such as:

  + Adding global namespace qualifiers so that corrections can refer to shadowed
    or otherwise ambiguous or unreachable namespaces.
  + Including accessible class members in the set of typo correction candidates,
    so that corrections requiring a class name in the name specifier are now
    possible.
  + Allowing typo corrections that involve removing a name specifier.
  + In some situations, correcting function names when a function was given the
    wrong number of arguments, including situations where the original function
    name was correct but was shadowed by a lexically closer function with the
    same name yet took a different number of arguments.
  + Offering typo suggestions for 'using' declarations.
  + Providing better diagnostics and fixit suggestions in more situations when
    a '->' was used instead of '.' or vice versa.
  + Providing more relevant suggestions for typos followed by '.' or '='.
  + Various performance improvements when searching for typo correction
    candidates.

New Compiler Flags
------------------

- Clang no longer special cases -O4 to enable lto. Explicitly pass -flto to
  enable it.
- Clang no longer fails on >= -O5. These flags are mapped to -O3 instead.
- Command line "clang -O3 -flto a.c -c" and "clang -emit-llvm a.c -c"
  are no longer equivalent.
- Clang now errors on unknown -m flags (``-munknown-to-clang``),
  unknown -f flags (``-funknown-to-clang``) and unknown
  options (``-what-is-this``).

C Language Changes in Clang
---------------------------

- Added new checked arithmetic builtins for security critical applications.

C11 Feature Support
^^^^^^^^^^^^^^^^^^^

...

C++ Language Changes in Clang
-----------------------------

- Fixed an ABI regression, introduced in Clang 3.2, which affected
  member offsets for classes inheriting from certain classes with tail padding.
  See PR16537.

- Clang 3.4 supports the 2013-08-28 draft of the ISO WG21 SG10 feature test
  macro recommendations. These aim to provide a portable method to determine
  whether a compiler supports a language feature, much like Clang's
  |has_feature macro|_.

.. |has_feature macro| replace:: ``__has_feature`` macro
.. _has_feature macro: LanguageExtensions.html#has-feature-and-has-extension

C++1y Feature Support
^^^^^^^^^^^^^^^^^^^^^

Clang 3.4 supports all the features in the current working draft of the
upcoming C++ standard, provisionally named C++1y. Support for the following
major new features has been added since Clang 3.3:

- Generic lambdas and initialized lambda captures.
- Deduced function return types (``auto f() { return 0; }``).
- Generalized ``constexpr`` support (variable mutation and loops).
- Variable templates and static data member templates.
- Use of ``'`` as a digit separator in numeric literals.
- Support for sized ``::operator delete`` functions.

In addition, ``[[deprecated]]`` is now accepted as a synonym for Clang's
existing ``deprecated`` attribute.

Use ``-std=c++1y`` to enable C++1y mode.

Objective-C Language Changes in Clang
-------------------------------------

...

OpenCL C Language Changes in Clang
----------------------------------

- OpenCL C "long" now always has a size of 64 bit, and all OpenCL C
  types are aligned as specified in the OpenCL C standard. Also,
  "char" is now always signed.

Internal API Changes
--------------------

These are major API changes that have happened since the 3.3 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

Wide Character Types
^^^^^^^^^^^^^^^^^^^^

The ASTContext class now keeps track of two different types for wide character
types: WCharTy and WideCharTy. WCharTy represents the built-in wchar_t type
available in C++. WideCharTy is the type used for wide character literals; in
C++ it is the same as WCharTy, but in C99, where wchar_t is a typedef, it is an
integer type.

...

libclang
--------

...

Static Analyzer
---------------

The static analyzer (which contains additional code checking beyond compiler
warnings) has improved significantly in both in the core analysis engine and 
also in the kinds of issues it can find.

Clang Format
------------

Clang now includes a new tool ``clang-format`` which can be used to
automatically format C, C++ and Objective-C source code. ``clang-format``
automatically chooses linebreaks and indentation and can be easily integrated
into editors, IDEs and version control systems. It supports several pre-defined
styles as well as precise style control using a multitude of formatting
options. ``clang-format`` itself is just a thin wrapper around a library which
can also be used directly from code refactoring and code translation tools.
More information can be found on `Clang Format's
site <http://clang.llvm.org/docs/ClangFormat.html>`_.

Windows Support
---------------

- `clang-cl <UsersManual.html#clang-cl>` provides a new driver mode that is
  designed for compatibility with Visual Studio's compiler, cl.exe. This driver
  mode makes Clang accept the same kind of command-line options as cl.exe. The
  installer will attempt to expose clang-cl in any Visual Studio installations
  on the system as a Platform Toolset, e.g. "LLVM-vs2012". clang-cl targets the
  Microsoft ABI by default. Please note that this driver mode and compatibility
  with the MS ABI is highly experimental.

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
list <http://lists.cs.uiuc.edu/mailman/listinfo/cfe-dev>`_.
