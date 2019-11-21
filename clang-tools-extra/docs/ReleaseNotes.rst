====================================================
Extra Clang Tools 10.0.0 (In-Progress) Release Notes
====================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 10 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 10.0.0. Here we describe the status of the Extra Clang Tools in
some detail, including major improvements from the previous release and new
feature work. All LLVM releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or
the `LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Subversion checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Extra Clang Tools 10.0.0?
=======================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

...

Improvements to clangd
----------------------

The improvements are...

Improvements to clang-doc
-------------------------

- :doc:`clang-doc <clang-doc>` now generates documentation in HTML format.

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- New :doc:`bugprone-bad-signal-to-kill-thread
  <clang-tidy/checks/bugprone-bad-signal-to-kill-thread>` check.

  Finds ``pthread_kill`` function calls when a thread is terminated by 
  raising ``SIGTERM`` signal.

- New :doc:`bugprone-dynamic-static-initializers
  <clang-tidy/checks/bugprone-dynamic-static-initializers>` check.

  Finds instances where variables with static storage are initialized
  dynamically in header files.

- New :doc:`bugprone-infinite-loop
  <clang-tidy/checks/bugprone-infinite-loop>` check.

  Finds obvious infinite loops (loops where the condition variable is not
  changed at all).

- New :doc:`bugprone-not-null-terminated-result
  <clang-tidy/checks/bugprone-not-null-terminated-result>` check

  Finds function calls where it is possible to cause a not null-terminated
  result. Usually the proper length of a string is ``strlen(str) + 1`` or equal
  length of this expression, because the null terminator needs an extra space.
  Without the null terminator it can result in undefined behaviour when the
  string is read.

- New :doc:`cert-mem57-cpp
  <clang-tidy/checks/cert-mem57-cpp>` check.

  Checks if an object of type with extended alignment is allocated by using
  the default ``operator new``.

- New alias :doc:`cert-pos44-c
  <clang-tidy/checks/cert-pos44-c>` to
  :doc:`bugprone-bad-signal-to-kill-thread
  <clang-tidy/checks/bugprone-bad-signal-to-kill-thread>` was added.

- New :doc:`cppcoreguidelines-init-variables
  <clang-tidy/checks/cppcoreguidelines-init-variables>` check.

- New :doc:`darwin-dispatch-once-nonstatic
  <clang-tidy/checks/darwin-dispatch-once-nonstatic>` check.

  Finds declarations of ``dispatch_once_t`` variables without static or global
  storage.

- New :doc:`google-upgrade-googletest-case
  <clang-tidy/checks/google-upgrade-googletest-case>` check.

  Finds uses of deprecated Googletest APIs with names containing ``case`` and
  replaces them with equivalent APIs with ``suite``.

- Improved :doc:`hicpp-signed-bitwise
  <clang-tidy/checks/hicpp-signed-bitwise>` check.

  The check now supports the ``IgnorePositiveIntegerLiterals`` option.

- New :doc:`linuxkernel-must-use-errs
  <clang-tidy/checks/linuxkernel-must-use-errs>` check.

  Checks Linux kernel code to see if it uses the results from the functions in
  ``linux/err.h``.

- New :doc:`llvm-prefer-register-over-unsigned
  <clang-tidy/checks/llvm-prefer-register-over-unsigned>` check.

  Finds historical use of ``unsigned`` to hold vregs and physregs and rewrites
  them to use ``Register``

- New :doc:`objc-missing-hash
  <clang-tidy/checks/objc-missing-hash>` check.

  Finds Objective-C implementations that implement ``-isEqual:`` without also
  appropriately implementing ``-hash``.

- New :doc:`performance-trivially-destructible
  <clang-tidy/checks/performance-trivially-destructible>` check.

  Finds types that could be made trivially-destructible by removing out-of-line
  defaulted destructor declarations.

- Improved :doc:`bugprone-posix-return
  <clang-tidy/checks/bugprone-posix-return>` check.

  Now also checks if any calls to ``pthread_*`` functions expect negative return
  values.

- The 'objc-avoid-spinlock' check was renamed to :doc:`darwin-avoid-spinlock
  <clang-tidy/checks/darwin-avoid-spinlock>`

- The :doc:`modernize-use-equals-default
  <clang-tidy/checks/modernize-use-equals-default>` fix no longer adds
  semicolons where they would be redundant.

- New :doc:`readability-redundant-access-specifiers
  <clang-tidy/checks/readability-redundant-access-specifiers>` check.

  Finds classes, structs, and unions that contain redundant member
  access specifiers.

- New :doc:`readability-make-member-function-const
  <clang-tidy/checks/readability-make-member-function-const>` check.

  Finds non-static member functions that can be made ``const``
  because the functions don't use ``this`` in a non-const way.

- Improved :doc:`modernize-use-override
  <clang-tidy/checks/modernize-use-override>` check.

  The check now supports the ``AllowOverrideAndFinal`` option to eliminate
  conflicts with ``gcc -Wsuggest-override`` or ``gcc -Werror=suggest-override``.

- Improved :doc:`readability-redundant-member-init
  <clang-tidy/checks/readability-redundant-member-init>` check.

  The check  now supports the ``IgnoreBaseInCopyConstructors`` option to avoid
  `"base class 'Foo' should be explicitly initialized in the copy constructor"`
  warnings or errors with ``gcc -Wextra`` or ``gcc -Werror=extra``.

- The :doc:`readability-redundant-string-init
  <clang-tidy/checks/readability-redundant-string-init>` check now supports a
  `StringNames` option enabling its application to custom string classes.

Improvements to include-fixer
-----------------------------

The improvements are...

Improvements to clang-include-fixer
-----------------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...

Improvements to pp-trace
------------------------

The improvements are...

Clang-tidy visual studio plugin
-------------------------------

The clang-tidy-vs plugin has been removed from clang, as
it's no longer maintained. Users should migrate to
`Clang Power Tools <https://marketplace.visualstudio.com/items?itemName=caphyon.ClangPowerTools>`_
instead.
