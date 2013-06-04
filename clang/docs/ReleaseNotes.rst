=======================
Clang 3.3 Release Notes
=======================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 3.3. Here we
describe the status of Clang in some detail, including major improvements from
the previous release and new feature work. For the general LLVM release notes,
see `the LLVM documentation <http://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web site
<http://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the latest
release, please check out the main please see the `Clang Web Site
<http://clang.llvm.org>`_ or the `LLVM Web Site <http://llvm.org>`_.

Note that if you are reading this file from a Subversion checkout or the main
Clang web page, this document applies to the *next* release, not the current
one. To see the release notes for a specific release, please see the `releases
page <http://llvm.org/releases/>`_.

What's New in Clang 3.3?
========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific sections with
improvements to Clang's support for those languages.

Major New Features
------------------

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang's diagnostics are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate source information
about them. The improvements since the 3.2 release include:

Extended Identifiers: Unicode Support and Universal Character Names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang 3.3 includes support for *extended identifiers* in C99 and C++.
This feature allows identifiers to contain certain Unicode characters, as
specified by the active language standard; these characters can be written
directly in the source file using the UTF-8 encoding, or referred to using
*universal character names* (``\u00E0``, ``\U000000E0``).

C Language Changes in Clang
---------------------------

C++ Language Changes in Clang
-----------------------------

- Clang now correctly implements language linkage for functions and variables.
  This means that, for example, it is now possible to overload static functions
  declared in an ``extern "C"`` context. For backwards compatibility, an alias
  with the unmangled name is still emitted if it is the only one and has the
  ``used`` attribute.

Internal API Changes
--------------------

These are major API changes that have happened since the 3.2 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

Value Casting
^^^^^^^^^^^^^

Certain type hierarchies (TypeLoc, CFGElement, ProgramPoint, and SVal) were
misusing the llvm::cast machinery to perform undefined operations. Their APIs
have been changed to use two member function templates that return values
instead of pointers or references - "T castAs" and "Optional<T> getAs" (in the
case of the TypeLoc hierarchy the latter is "T getAs" and you can use the
boolean testability of a TypeLoc (or its 'validity') to verify that the cast
succeeded). Essentially all previous 'cast' usage should be replaced with
'castAs' and 'dyn_cast' should be replaced with 'getAs'. See r175462 for the
first example of such a change along with many examples of how code was
migrated to the new API.

Storage Class
^^^^^^^^^^^^^

For each variable and function Clang used to keep the storage class as written
in the source, the linkage and a semantic storage class. This was a bit
redundant and the semantic storage class has been removed. The method
getStorageClass now returns what is written in the source code for that decl.

Wide Character Types
^^^^^^^^^^^^^^^^^^^^

The ASTContext class now keeps track of two different types for wide character
types: WCharTy and WideCharTy. WCharTy represents the built-in wchar_t type
available in C++. WideCharTy is the type used for wide character literals; in
C++ it is the same as WCharTy, but in C99, where wchar_t is a typedef, it is an
integer type.

libclang
--------

The clang_CXCursorSet_contains() function previously incorrectly returned 0
if it contained a CXCursor, contrary to what the documentation stated.  This
has been fixed so that the function returns a non-zero value if the set
contains a cursor.  This is API breaking change, but matches the intended
original behavior.  Moreover, this also fixes the issue of an invalid CXCursorSet
appearing to contain any CXCursor.

Static Analyzer
---------------

The static analyzer (which contains additional code checking beyond compiler
warnings) has improved significantly in both in the core analysis engine and 
also in the kinds of issues it can find.

Core Analysis Improvements
==========================

- Support for interprocedural reasoning about constructors and destructors.
- New false positive suppression mechanisms that reduced the number of false
  null pointer dereference warnings due to interprocedural analysis.
- Major performance enhancements to speed up interprocedural analysis

New Issues Found
================

- New memory error checks such as use-after-free with C++ 'delete'.
- Detection of mismatched allocators and deallocators (e.g., using 'new' with
  'free()', 'malloc()' with 'delete').
- Additional checks for misuses of Apple Foundation framework collection APIs.

Significant Known Problems
==========================

Additional Information
======================

A wide variety of additional information is available on the `Clang web page
<http://clang.llvm.org/>`_. The web page contains versions of the API
documentation which are up-to-date with the Subversion version of the source
code. You can access versions of these documents specific to this release by
going into the "``clang/docs/``" directory in the Clang tree.

If you have any questions or comments about Clang, please feel free to contact
us via the `mailing list <http://lists.cs.uiuc.edu/mailman/listinfo/cfe-dev>`_.
