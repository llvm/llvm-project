===================================
Extra Clang Tools 3.9 Release Notes
===================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <http://llvm.org/>`_

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

What's New in Extra Clang Tools 3.9?
====================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

- :program:`clang-include-fixer`, a tool that provides an automated way of
  adding ``#include`` directives for missing symbols in one translation unit.

  It aims to provide automated insertion of missing ``#includes`` with a single
  button press in an editor. Integration with Vim and a tool to generate the
  symbol index used by the tool are also part of this release. See the
  `include-fixer documentation`_ for more information.

.. _include-fixer documentation: http://clang.llvm.org/extra/include-fixer.html

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

- New Boost module containing checks for issues with Boost library.

- New `boost-use-to-string 
  <http://clang.llvm.org/extra/clang-tidy/checks/boost-use-to-string.html>`_ check

  Finds usages of ``boost::lexical_cast<std::string>`` and changes it to
  ``std::to_string``.

- New `cert-env33-c
  <http://clang.llvm.org/extra/clang-tidy/checks/cert-env33-c.html>`_ check

  Flags calls to ``system()``, ``popen()``, and ``_popen()``, which execute a
  command processor.

- New `cert-err34-c
  <http://clang.llvm.org/extra/clang-tidy/checks/cert-err34-c.html>`_ check

  Flags calls to string-to-number conversion functions that do not verify the
  validity of the conversion.

- New `cert-flp30-c
  <http://clang.llvm.org/extra/clang-tidy/checks/cert-flp30-c.html>`_ check

  Flags ``for`` loops where the induction expression has a floating-point type.

- New `cppcoreguidelines-interfaces-global-init
  <http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-interfaces-global-init.html>`_ check

  Flags initializers of globals that access extern objects, and therefore can
  lead to order-of-initialization problems.

- New `cppcoreguidelines-pro-type-member-init
  <http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-pro-type-member-init.html>`_ check

  Flags user-defined constructor definitions that do not initialize all builtin
  and pointer fields which leaves their memory in an undefined state.

- New `google-default-arguments
  <http://clang.llvm.org/extra/clang-tidy/checks/google-default-arguments.html>`_ check

  Flags default arguments in virtual methods.

- New `misc-dangling-handle
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-dangling-handle.html>`_ check

  Detects dangling references in value handlers like
  ``std::experimental::string_view``.

- New `misc-fold-init-type
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-fold-init-type.html>`_ check

  The check flags type mismatches in `folds` like ``std::accumulate`` that might
  result in loss of precision.

- New `misc-forward-declaration-namespace
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-forward-declaration-namespace.html>`_ check

  Checks if an unused forward declaration is in a wrong namespace.

- New `misc-misplaced-const
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-misplaced-const.html>`_ check
  
  Checks if a ``const`` qualifier is applied to a ``typedef`` to pointer type
  instead of the underlying pointee type.

- New `misc-misplaced-widening-cast
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-misplaced-widening-cast.html>`_ check

  Warns when there is a explicit redundant cast of a calculation result to a
  bigger type.

- New `misc-multiple-statement-macro
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-multiple-statement-macro.html>`_ check

  Detect multiple statement macros that are used in unbraced conditionals.

- New `misc-pointer-and-integral-operation
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-pointer-and-integral-operation.html>`_ check

  Warns about suspicious operations involving pointers and integral types.

- New `misc-redundant-expression
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-redundant-expression.html>`_ check

  Warns about redundant and equivalent expressions.

- New `misc-sizeof-expression
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-sizeof-expression.html>`_ check

  Warns about incorrect uses of ``sizeof`` operator.

- New `misc-string-constructor
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-string-constructor.html>`_ check

  Finds string constructors that are suspicious and probably errors.

- New `misc-string-literal-with-embedded-nul
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-string-literal-with-embedded-nul.html>`_ check

  Warns about suspicious NUL character in string literals which may lead to
  truncation or invalid character escaping.

- New `misc-suspicious-missing-comma
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-suspicious-missing-comma.html>`_ check

  Warns about 'probably' missing comma in string literals initializer list.

- New `misc-suspicious-semicolon
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-suspicious-semicolon.html>`_ check

  Finds most instances of stray semicolons that unexpectedly alter the meaning
  of the code.

- New `misc-suspicious-string-compare
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-suspicious-string-compare.html>`_ check

  Find suspicious usage of runtime string comparison functions.

- New `misc-unconventional-assign-operator
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-unconventional-assign-operator.html>`_
  check replacing the *misc-assign-operator-signature* check.

  Does not only checks for correct signature but also for correct ``return``
  statements (returning ``*this``)

- New `misc-unused-using-decls
  <http://clang.llvm.org/extra/clang-tidy/checks/misc-unused-using-decls.html>`_ check

  Finds unused ``using`` declarations.

- New `modernize-avoid-bind
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-avoid-bind.html>`_ check

  Finds uses of ``std::bind`` and replaces simple uses with lambdas.

- New `modernize-deprecated-headers
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-deprecated-headers.html>`_ check

  Replaces C standard library headers with their C++ alternatives.

- New `modernize-make-shared
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-make-shared.html>`_ check

  Replaces creation of ``std::shared_ptr`` from new expression with call to ``std::make_shared``.

- New `modernize-raw-string-literal
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-raw-string-literal.html>`_ check

  Selectively replaces string literals containing escaped characters with raw
  string literals.

- New `modernize-use-bool-literals
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-bool-literals.html>`_ check

  Finds integer literals which are cast to ``bool``.

- New `modernize-use-emplace
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-emplace.html>`_ check

  Finds calls that could be changed to emplace.

- New `modernize-use-using
  <http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-using.html>`_ check

  Finds typedefs and replaces it with usings.

- New `performance-faster-string-find
  <http://clang.llvm.org/extra/clang-tidy/checks/performance-faster-string-find.html>`_ check

  Optimize calls to ``std::string::find()`` and friends when the needle passed
  is a single character string literal.

- New `performance-implicit-cast-in-loop
  <http://clang.llvm.org/extra/clang-tidy/checks/performance-implicit-cast-in-loop.html>`_ check

  Warns about range-based loop with a loop variable of const ref type where the
  type of the variable does not match the one returned by the iterator.

- New `performance-unnecessary-value-param
  <http://clang.llvm.org/extra/clang-tidy/checks/performance-unnecessary-value-param.html>`_ check

  Flags value parameter declarations of expensive to copy types that are copied
  for each invocation but it would suffice to pass them by const reference.

- New `readability-avoid-const-params-in-decls
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-avoid-const-params-in-decls.html>`_ check

  Warns about top-level const parameters in function declarations.

- New `readability-deleted-default
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-deleted-default.html>`_ check

  Warns about defaulted constructors and assignment operators that are actually
  deleted.

- Updated `readability-identifier-naming-check
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-identifier-naming.html>`_

  Added support for enforcing the case of macro statements.

- New `readability-redundant-control-flow
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-control-flow.html>`_ check

  Looks for procedures (functions returning no value) with ``return`` statements
  at the end of the function.  Such `return` statements are redundant.

- New `readability-redundant-string-init
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-string-init.html>`_ check

  Finds unnecessary string initializations.

- New `readability-static-definition-in-anonymous-namespace
  <http://clang.llvm.org/extra/clang-tidy/checks/readability-static-definition-in-anonymous-namespace.html>`_ check

  Finds static function and variable definitions in anonymous namespace.

Fixed bugs:

- Crash when running on compile database with relative source files paths.

- Crash when running with the `-fdelayed-template-parsing` flag.

- The `modernize-use-override` check: incorrect fix-its placement around
  ``__declspec`` and other attributes.

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

- Updated ``cppcoreguidelines-pro-member-type-member-init`` check

  This check now conforms to C++ Core Guidelines rule Type.6: Always Initialize
  a Member Variable. The check examines every record type where construction
  might result in an undefined memory state. These record types needing
  initialization have at least one default-initialized built-in, pointer,
  array or record type matching these criteria or a default-initialized
  direct base class of this kind.

  The check has two complementary aspects:

  1. Ensure every constructor for a record type needing initialization
     value-initializes all members and direct bases via a combination of
     in-class initializers and the member initializer list.
  2. Value-initialize every non-member instance of a record type needing
     initialization that lacks a user-provided default constructor, e.g.
     a POD.

Improvements to modularize
--------------------------

The improvements are...
