===================================================
Extra Clang Tools 7.0.0 (In-Progress) Release Notes
===================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 7 release.
   Release notes for previous releases can be found on
   `the Download Page <http://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 7.0.0. Here we describe the status of the Extra Clang Tools in
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

What's New in Extra Clang Tools 7.0.0?
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

The improvements are...

Improvements to clang-tidy
--------------------------

- New `cppcoreguidelines-avoid-goto
  <http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-avoid-goto.html>`_ check

  The usage of ``goto`` for control flow is error prone and should be replaced
  with looping constructs. Every backward jump is rejected. Forward jumps are
  only allowed in nested loops.

- New `fuchsia-multiple-inheritance
  <http://clang.llvm.org/extra/clang-tidy/checks/fuchsia-multiple-inheritance.html>`_ check

  Warns if a class inherits from multiple classes that are not pure virtual.

- New `fuchsia-statically-constructed-objects
  <http://clang.llvm.org/extra/clang-tidy/checks/fuchsia-statically-constructed-objects.html>`_ check

  Warns if global, non-trivial objects with static storage are constructed, unless the 
  object is statically initialized with a ``constexpr`` constructor or has no 
  explicit constructor.
  
- New `fuchsia-trailing-return
  <http://clang.llvm.org/extra/clang-tidy/checks/fuchsia-trailing-return.html>`_ check

  Functions that have trailing returns are disallowed, except for those 
  using decltype specifiers and lambda with otherwise unutterable 
  return types.
    
- New alias `hicpp-avoid-goto
  <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-avoid-goto.html>`_ to 
  `cppcoreguidelines-avoid-goto <http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-avoid-goto.html>`_
  added.

Improvements to include-fixer
-----------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...
