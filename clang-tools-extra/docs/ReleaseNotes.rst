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

- New `google-avoid-throwing-objc-exception
  <http://clang.llvm.org/extra/clang-tidy/checks/google-objc-avoid-throwing-exception.html>`_ check

  Add new check to detect throwing exceptions in Objective-C code, which should be avoided.

- New `objc-property-declaration
  <http://clang.llvm.org/extra/clang-tidy/checks/objc-property-declaration.html>`_ check

  Add new check for Objective-C code to ensure property
  names follow the naming convention of Apple's programming
  guide.

- New `google-objc-global-variable-declaration
  <http://clang.llvm.org/extra/clang-tidy/checks/google-global-variable-declaration.html>`_ check

  Add new check for Objective-C code to ensure global 
  variables follow the naming convention of 'k[A-Z].*' (for constants) 
  or 'g[A-Z].*' (for variables).

- New module `objc` for Objective-C style checks.

- New `objc-forbidden-subclassing
  <http://clang.llvm.org/extra/clang-tidy/checks/objc-forbidden-subclassing.html>`_ check

  Ensures Objective-C classes do not subclass any classes which are
  not intended to be subclassed. Includes a list of classes from Foundation
  and UIKit which are documented as not supporting subclassing.

- New `bugprone-misplaced-operator-in-strlen-in-alloc
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-misplaced-operator-in-strlen-in-alloc.html>`_ check

  Finds cases where ``1`` is added to the string in the argument to
  ``strlen()``, ``strnlen()``, ``strnlen_s()``, ``wcslen()``, ``wcsnlen()``, and
  ``wcsnlen_s()`` instead of the result and the value is used as an argument to
  a memory allocation function (``malloc()``, ``calloc()``, ``realloc()``,
  ``alloca()``).

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

- New `bugprone-copy-constructor-init
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-copy-constructor-init.html>`_ check

  Finds copy constructors which don't call the copy constructor of the base class.

- New `bugprone-integer-division
  <http://clang.llvm.org/extra/clang-tidy/checks/bugprone-integer-division.html>`_ check

  Finds cases where integer division in a floating point context is likely to
  cause unintended loss of precision.

- New `cppcoreguidelines-owning-memory <http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-owning-memory.html>`_ check 

  This check implements the type-based semantic of ``gsl::owner<T*>``, but without
  flow analysis.

- New `hicpp-exception-baseclass
  <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-exception-baseclass.html>`_ check

  Ensures that all exception will be instances of ``std::exception`` and classes 
  that are derived from it.

- New `hicpp-signed-bitwise
  <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-signed-bitwise.html>`_ check

  Finds uses of bitwise operations on signed integer types, which may lead to 
  undefined or implementation defined behaviour.

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

- Added aliases for the `High Integrity C++ Coding Standard <http://www.codingstandard.com/section/index/>`_ 
  to already implemented checks in other modules.

  - `hicpp-deprecated-headers <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-deprecated-headers.html>`_
  - `hicpp-move-const-arg <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-move-const-arg.html>`_
  - `hicpp-no-array-decay <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-no-array-decay.html>`_
  - `hicpp-no-malloc <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-no-malloc.html>`_
  - `hicpp-static-assert <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-static-assert.html>`_
  - `hicpp-use-auto <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-use-auto.html>`_
  - `hicpp-use-emplace <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-use-emplace.html>`_
  - `hicpp-use-noexcept <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-use-noexcept.html>`_
  - `hicpp-use-nullptr <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-use-nullptr.html>`_
  - `hicpp-vararg <http://clang.llvm.org/extra/clang-tidy/checks/hicpp-vararg.html>`_

Improvements to include-fixer
-----------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...
