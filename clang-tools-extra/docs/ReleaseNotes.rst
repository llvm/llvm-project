===================================================
Extra Clang Tools 4.0.0 (In-Progress) Release Notes
===================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 4.0 release.
   You may prefer the `Extra Clang Tools 3.9 Release Notes
   <http://llvm.org/releases/3.9.0/tools/clang/tools/extra/docs/ReleaseNotes.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 4.0.0. Here we describe the status of the Extra Clang Tools in
some detail, including major improvements from the previous release and new
feature work. All LLVM releases may be downloaded from the `LLVM releases web
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

- `misc-pointer-and-integral-operation` check was removed.

- New `misc-use-after-move
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-use-after-move.html>`_ check

  Warns if an object is used after it has been moved, without an intervening
  reinitialization.

- New `cppcoreguidelines-no-malloc
  <http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-no-malloc.html>`_ check
  warns if C-style memory management is used and suggests the use of RAII.

- `modernize-make-unique
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-make-unique.html>`_
  and `modernize-make-shared
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-make-shared.html>`_
  now handle calls to the smart pointer's ``reset()`` method.

- The `modernize-use-auto
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-auto.html>`_ check
  now warns about variable declarations that are initialized with a cast, or by
  calling a templated function that behaves as a cast.

- The modernize-use-default check has been renamed to `modernize-use-equals-default
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-equals-default.html>`_.

- New `modernize-use-equals-delete
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-equals-delete.html>`_ check

  Adds ``= delete`` to unimplemented private special member functions.

- New `modernize-use-transparent-functors
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-transparent-functors.html>`_ check

  Replaces uses of non-transparent functors with transparent ones where applicable.

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

- New `performance-type-promotion-in-math-fn
  <http://clang.llvm.org/extra/clang-tidy/checks/performance-type-promotion-in-math-fn.html>`_ check

  Replaces uses of C-style standard math functions with double parameters and float
  arguments with an equivalent function that takes a float parameter.

- `readability-container-size-empty
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-container-size-empty.html>`_ check
  supports arbitrary containers with with suitable ``empty()`` and ``size()``
  methods.

- New `readability-misplaced-array-index
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-misplaced-array-index.html>`_ check

  Warns when there is array index before the [] instead of inside it.

- New `readability-non-const-parameter
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-non-const-parameter.html>`_ check

  Flags function parameters of a pointer type that could be changed to point to
  a constant type instead.

- New `readability-redundant-declaration
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-declaration.html>`_ check

  Finds redundant variable and function declarations.

- New `readability-redundant-function-ptr-dereference
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-function-ptr-dereference.html>`_ check

  Finds redundant function pointer dereferences.

- New `readability-redundant-member-init
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-member-init.html>`_ check

  Flags member initializations that are unnecessary because the same default
  constructor would be called if they were not present.

- The `readability-redundant-string-cstr
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-string-cstr.html>`_ check
  now warns about redundant calls to data() too.

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
