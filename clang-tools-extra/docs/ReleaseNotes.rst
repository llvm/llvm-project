=================================================
Extra Clang Tools 3.9 (In-Progress) Release Notes
=================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 3.9 release. You may
   prefer the `Clang 3.8 Release Notes
   <http://llvm.org/releases/3.8.0/tools/clang/docs/ReleaseNotes.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 3.9.  Here we describe the status of the Extra Clang Tools in some
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

What's New in Extra Clang Tools 3.9?
====================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

- Feature1...

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

:program:`clang-tidy`'s checks are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate fix-its for the issues
identified.  The improvements since the 3.8 release include:

- New ``modernize-raw-string-literal`` check

  selectively replaces string literals containing escaped characters with raw
  string literals.

- New ``readability-avoid-const-params-in-decls`` check

  warns about top-level const parameters in function declarations.

Clang-tidy changes from 3.7 to 3.8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The 3.8 release didn't include release notes for :program:`clang-tidy`. In the
3.8 release many new checks have been added to :program:`clang-tidy`:

- Checks enforcing certain rules of the `CERT Secure Coding Standards
  <https://www.securecoding.cert.org/confluence/display/seccode/SEI+CERT+Coding+Standards>`_:

  * `cert-dcl03-c
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-dcl03-c.html>`_
    (an alias to the pre-existing check `misc-static-assert
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-static-assert.html>`_)
  * `cert-dcl50-cpp
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-dcl50-cpp.html>`_
  * `cert-err52-cpp
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-err52-cpp.html>`_
  * `cert-err58-cpp
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-err58-cpp.html>`_
  * `cert-err60-cpp
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-err60-cpp.html>`_
  * `cert-err61-cpp
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-err61-cpp.html>`_
  * `cert-fio38-c
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-fio38-c.html>`_
    (an alias to the pre-existing check `misc-non-copyable-objects
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-non-copyable-objects.html>`_)
  * `cert-oop11-cpp
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cert-oop11-cpp.html>`_
    (an alias to the pre-existing check `misc-move-constructor-init
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/misc-move-constructor-init.html>`_)

- Checks supporting the `C++ Core Guidelines
  <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`_:

  * `cppcoreguidelines-pro-bounds-array-to-pointer-decay
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-bounds-array-to-pointer-decay.html>`_
  * `cppcoreguidelines-pro-bounds-constant-array-index
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-bounds-constant-array-index.html>`_
  * `cppcoreguidelines-pro-bounds-pointer-arithmetic
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-bounds-pointer-arithmetic.html>`_
  * `cppcoreguidelines-pro-type-const-cast
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-const-cast.html>`_
  * `cppcoreguidelines-pro-type-cstyle-cast
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-cstyle-cast.html>`_
  * `cppcoreguidelines-pro-type-reinterpret-cast
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-reinterpret-cast.html>`_
  * `cppcoreguidelines-pro-type-static-cast-downcast
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-static-cast-downcast.html>`_
  * `cppcoreguidelines-pro-type-union-access
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-union-access.html>`_
  * `cppcoreguidelines-pro-type-vararg
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-type-vararg.html>`_

- The functionality of the :program:`clang-modernize` tool has been moved to the
  new ``modernize`` module in :program:`clang-tidy` along with a few new checks:

  * `modernize-loop-convert
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-loop-convert.html>`_
  * `modernize-make-unique
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-make-unique.html>`_
  * `modernize-pass-by-value
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-pass-by-value.html>`_
  * `modernize-redundant-void-arg
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-redundant-void-arg.html>`_
  * `modernize-replace-auto-ptr
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-replace-auto-ptr.html>`_
  * `modernize-shrink-to-fit
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-shrink-to-fit.html>`_
    (renamed from ``readability-shrink-to-fit``)
  * `modernize-use-auto
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-use-auto.html>`_
  * `modernize-use-default
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-use-default.html>`_
  * `modernize-use-nullptr
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-use-nullptr.html>`_
  * `modernize-use-override
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/modernize-use-override.html>`_
    (renamed from ``misc-use-override``)

- New checks flagging various readability-related issues:

  * `readability-identifier-naming
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/readability-identifier-naming.html>`_
  * `readability-implicit-bool-cast
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/readability-implicit-bool-cast.html>`_
  * `readability-inconsistent-declaration-parameter-name
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/readability-inconsistent-declaration-parameter-name.html>`_
  * `readability-uniqueptr-delete-release
    <http://llvm.org/releases/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/readability-uniqueptr-delete-release.html>`_


Improvements to modularize
--------------------------

The improvements are...
