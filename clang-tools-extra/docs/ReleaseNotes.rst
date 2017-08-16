===================================================
Extra Clang Tools 6.0.0 (In-Progress) Release Notes
===================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 6 release.
   Release notes for previous releases can be found on
   `the Download Page <http://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 6.0.0. Here we describe the status of the Extra Clang Tools in
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

What's New in Extra Clang Tools 6.0.0?
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

- Renamed checks to use correct term "implicit conversion" instead of "implicit
  cast" and modified messages and option names accordingly:

    * **performance-implicit-cast-in-loop** was renamed to
      `performance-implicit-conversion-in-loop
      <http://clang.llvm.org/extra/clang-tidy/checks/performance-implicit-conversion-in-loop.html>`_
    * **readability-implicit-bool-cast** was renamed to
      `readability-implicit-bool-conversion
      <http://clang.llvm.org/extra/clang-tidy/checks/readability-implicit-bool-conversion.html>`_;
      the check's options were renamed as follows:
      ``AllowConditionalIntegerCasts`` -> ``AllowIntegerConditions``,
      ``AllowConditionalPointerCasts`` -> ``AllowPointerConditions``.

- New `android-cloexec-accept
  <http://clang.llvm.org/extra/clang-tidy/checks/android-cloexec-accept.html>`_ check

  Detects usage of ``accept()``.

- New `android-cloexec-accept4
  <http://clang.llvm.org/extra/clang-tidy/checks/android-cloexec-accept4.html>`_ check

  Checks if the required file flag ``SOCK_CLOEXEC`` is present in the argument of
  ``accept4()``.

- New `android-cloexec-dup
  <http://clang.llvm.org/extra/clang-tidy/checks/android-cloexec-dup.html>`_ check

  Detects usage of ``dup()``.

- New `android-cloexec-inotify-init
  <http://clang.llvm.org/extra/clang-tidy/checks/android-cloexec-inotify-init.html>`_ check

  Detects usage of ``inotify_init()``.

- New `android-cloexec-epoll-create1
  <http://clang.llvm.org/extra/clang-tidy/checks/android-cloexec-epoll-create1.html>`_ check

  Checks if the required file flag ``EPOLL_CLOEXEC`` is present in the argument of
  ``epoll_create1()``.

- New `android-cloexec-epoll-create
  <http://clang.llvm.org/extra/clang-tidy/checks/android-cloexec-epoll-create.html>`_ check

  Detects usage of ``epoll_create()``.

- New `android-cloexec-memfd_create
  <http://clang.llvm.org/extra/clang-tidy/checks/android-cloexec-memfd_create.html>`_ check

  Checks if the required file flag ``MFD_CLOEXEC`` is present in the argument
  of ``memfd_create()``.

- New `bugprone-integer-division
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-integer-division.html>`_ check

  Finds cases where integer division in a floating point context is likely to
  cause unintended loss of precision.

- New `hicpp-exception-baseclass
  <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-exception-baseclass.html>`_ check

  Ensures that all exception will be instances of ``std::exception`` and classes 
  that are derived from it.

- New `android-cloexec-inotify-init1
  <http://clang.llvm.org/extra/clang-tidy/checks/android-cloexec-inotify-init1.html>`_ check

  Checks if the required file flag ``IN_CLOEXEC`` is present in the argument of
  ``inotify_init1()``.

- New `readability-static-accessed-through-instance
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-static-accessed-through-instance.html>`_ check

  Finds member expressions that access static members through instances and
  replaces them with uses of the appropriate qualified-id.

- Added `modernize-use-emplace.IgnoreImplicitConstructors
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-emplace.html#cmdoption-arg-IgnoreImplicitConstructors>`_
  option.

- Added alias `hicpp-braces-around-statements <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-braces-around-statements.html>`_ 

Improvements to include-fixer
-----------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...
