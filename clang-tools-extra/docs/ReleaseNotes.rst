===================================================
Extra Clang Tools 4.0.0 (In-Progress) Release Notes
===================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 4.0.0 release. You may
   prefer the `Clang 3.8 Release Notes
   <http://llvm.org/releases/3.8.0/tools/clang/docs/ReleaseNotes.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 4.0.0.  Here we describe the status of the Extra Clang Tools in some
detail, including major improvements from the previous release and new feature
work. For the general Clang release notes, see `the Clang documentation
<http://llvm.org/releases/3.8.0/tools/clang/docs/ReleaseNotes.html>`_.  All LLVM
releases may be downloaded from the `LLVM releases web
site <http://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please see the `Clang Web Site <http://clang.llvm.org>`_ or
the `LLVM Web Site <http://llvm.org>`_.

Note that if you are reading this file from a Subversion checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <http://llvm.org/releases/>`_.

What's New in Extra Clang Tools 4.0.0?
======================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

...

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

- Emacs integration was added.

Improvements to clang-tidy
--------------------------

- New `cppcoreguidelines-slicing
  <http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-slicing.html>`_ check

  Flags slicing of member variables or vtable.

- New `cppcoreguidelines-special-member-functions
  <http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-special-member-functions.html>`_ check

  Flags classes where some, but not all, special member functions are user-defined.

- New `misc-move-forwarding-reference
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-move-forwarding-reference.html>`_ check

  Warns when ``std::move`` is applied to a forwarding reference instead of
  ``std::forward``.

- New `mpi-buffer-deref
  <http://clang.llvm.org/extra/clang-tidy/checks/mpi-buffer-deref.html>`_ check

  Flags buffers which are insufficiently dereferenced when passed to an MPI function call.

- New `mpi-type-mismatch
  <http://clang.llvm.org/extra/clang-tidy/checks/mpi-type-mismatch.html>`_ check

  Flags MPI function calls with a buffer type and MPI data type mismatch.

- New `performance-inefficient-string-concatenation
  <http://clang.llvm.org/extra/clang-tidy/checks/performance-inefficient-string-concatenation.html>`_ check

  Warns about the performance overhead arising from concatenating strings using
  the ``operator+``, instead of ``operator+=``.

- New `readability-non-const-parameter
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-non-const-parameter.html>`_ check

  Flags function parameters of a pointer type that could be changed to point to
  a constant type instead.

Fixed bugs:

- `modernize-make-unique
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-make-unique.html>`_
  and `modernize-make-shared
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-make-shared.html>`_
  Calling ``make_{unique|shared}`` from within a member function of a type
  with a private or protected constructor would be ill-formed.

Improvements to include-fixer
-----------------------------

- Emacs integration was added.

Improvements to modularize
--------------------------

The improvements are...
