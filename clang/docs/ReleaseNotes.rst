=======================================
Clang 5.0.0 (In-Progress) Release Notes
=======================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 5 release.
   Release notes for previous releases can be found on
   `the Download Page <http://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 5.0.0. Here we
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

What's New in Clang 5.0.0?
==========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

-  ...

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  -Wunused-lambda-capture warns when a variable explicitly captured
   by a lambda is not used in the body of the lambda.

New Compiler Flags
------------------

The option ....

New Pragmas in Clang
-----------------------

Clang now supports the ...


Attribute Changes in Clang
--------------------------

-  ...

Windows Support
---------------

Clang's support for building native Windows programs ...


C Language Changes in Clang
---------------------------

- ...

...

C11 Feature Support
^^^^^^^^^^^^^^^^^^^

...

C++ Language Changes in Clang
-----------------------------

...

C++1z Feature Support
^^^^^^^^^^^^^^^^^^^^^

...

Objective-C Language Changes in Clang
-------------------------------------

...

OpenCL C Language Changes in Clang
----------------------------------

...

OpenMP Support in Clang
----------------------------------

...

Internal API Changes
--------------------

These are major API changes that have happened since the 4.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

-  ...

AST Matchers
------------

...


clang-format
------------

* Option **BreakBeforeInheritanceComma** added to break before ``:`` and ``,``  in case of
  multiple inheritance in a class declaration. Enabled by default in the Mozilla coding style.

  +---------------------+----------------------------------------+
  | true                | false                                  |
  +=====================+========================================+
  | .. code-block:: c++ | .. code-block:: c++                    |
  |                     |                                        |
  |   class MyClass     |   class MyClass : public X, public Y { |
  |       : public X    |   };                                   |
  |       , public Y {  |                                        |
  |   };                |                                        |
  +---------------------+----------------------------------------+

* Align block comment decorations.

  +----------------------+---------------------+
  | Before               | After               |
  +======================+=====================+
  |  .. code-block:: c++ | .. code-block:: c++ |
  |                      |                     |
  |    /* line 1         |   /* line 1         |
  |      * line 2        |    * line 2         |
  |     */               |    */               |
  +----------------------+---------------------+

* The :doc:`ClangFormatStyleOptions` documentation provides detailed examples for most options.

* Namespace end comments are now added or updated automatically.

  +---------------------+---------------------+
  | Before              | After               |
  +=====================+=====================+
  | .. code-block:: c++ | .. code-block:: c++ |
  |                     |                     |
  |   namespace A {     |   namespace A {     |
  |   int i;            |   int i;            |
  |   int j;            |   int j;            |
  |   }                 |   }                 |
  +---------------------+---------------------+

* Comment reflow support added. Overly long comment lines will now be reflown with the rest of
  the paragraph instead of just broken. Option **ReflowComments** added and enabled by default.

libclang
--------

...


Static Analyzer
---------------

...

Undefined Behavior Sanitizer (UBSan)
------------------------------------

- The Undefined Behavior Sanitizer has a new check for pointer overflow. This
  check is on by default. The flag to control this functionality is
  -fsanitize=pointer-overflow.

  Pointer overflow is an indicator of undefined behavior: when a pointer
  indexing expression wraps around the address space, or produces other
  unexpected results, its result may not point to a valid object.

- UBSan has several new checks which detect violations of nullability
  annotations. These checks are off by default. The flag to control this group
  of checks is -fsanitize=nullability. The checks can be individially enabled
  by -fsanitize=nullability-arg (which checks calls),
  -fsanitize=nullability-assign (which checks assignments), and
  -fsanitize=nullability-return (which checks return statements).

- UBSan can now detect invalid loads from bitfields and from ObjC BOOLs.

- UBSan can now avoid emitting unnecessary type checks in C++ class methods and
  in several other cases where the result is known at compile-time. UBSan can
  also avoid emitting unnecessary overflow checks in arithmetic expressions
  with promoted integer operands.

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
