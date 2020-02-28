======================================
Extra Clang Tools 10.0.0 Release Notes
======================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

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

What's New in Extra Clang Tools 10.0.0?
=======================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.


Improvements to clangd
----------------------

- Go-to-definition, hover, find-references etc use a new mechanism to identify
  what is under the cursor, which is (hopefully) more consistent and accurate.

- clangd should be able to reliably locate the standard library/SDK on macOS.

- Shutdown more cleanly on receiving a signal. In particular temporary PCH files
  should be cleaned up.

- Find references now works on macros.

- clangd can be more easily used remotely or in a docker container.

  The ``--path-mappings`` flag translates between local and remote paths.

- Experimental support for renaming across files (behind the
  ``--cross-file-rename`` flag).

- Hover now exposes more information, including the type of symbols and the
  value of constant expressions.

- Go to definition now works in dependent code in more cases, by assuming the
  primary template is used.

- Better recovery and reporting when the compile command for a file can't be
  fully parsed.

- Switch header/source (an extension) now uses index information in addition
  to filename heuristics, and is much more robust.

- Semantic selection (expand/contract selection) is supported.

- Semantic highlighting is more robust, highlights more types of tokens, and
  as an extension provides information about inactive preprocessor regions.

- Code completion results now include an extension field ``score``.

  This allows clients to incorporate clangd quality signals when re-ranking code
  completion after client-side fuzzy-matching.

- New refactorings:
  define function out-of-line, define function in-line, extract function,
  remove using namespace directive, localize Objective-C string.

- Bug fixes and performance improvements :-)

Improvements to clang-doc
-------------------------

- :doc:`clang-doc <clang-doc>` now generates documentation in HTML format.

Improvements to clang-tidy
--------------------------

New checks
^^^^^^^^^^

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
  result.

- New :doc:`bugprone-signed-char-misuse
  <clang-tidy/checks/bugprone-signed-char-misuse>` check.

  Finds ``signed char`` to integer conversions which might indicate a
  programming error.

- New :doc:`cert-mem57-cpp
  <clang-tidy/checks/cert-mem57-cpp>` check.

  Checks if an object of type with extended alignment is allocated by using
  the default ``operator new``.

- New :doc:`cert-oop58-cpp
  <clang-tidy/checks/cert-oop58-cpp>` check.

  Finds assignments to the copied object and its direct or indirect members
  in copy constructors and copy assignment operators.

- New :doc:`cppcoreguidelines-init-variables
  <clang-tidy/checks/cppcoreguidelines-init-variables>` check.

  Checks whether there are local variables that are declared without an initial
  value.

- New :doc:`darwin-dispatch-once-nonstatic
  <clang-tidy/checks/darwin-dispatch-once-nonstatic>` check.

  Finds declarations of ``dispatch_once_t`` variables without static or global
  storage.

- New :doc:`google-upgrade-googletest-case
  <clang-tidy/checks/google-upgrade-googletest-case>` check.

  Finds uses of deprecated Googletest APIs with names containing ``case`` and
  replaces them with equivalent APIs with ``suite``.

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

- New :doc:`performance-no-automatic-move
  <clang-tidy/checks/performance-no-automatic-move>` check.

  Finds local variables that cannot be automatically moved due to constness.

- New :doc:`performance-trivially-destructible
  <clang-tidy/checks/performance-trivially-destructible>` check.

  Finds types that could be made trivially-destructible by removing out-of-line
  defaulted destructor declarations.

- New :doc:`readability-make-member-function-const
  <clang-tidy/checks/readability-make-member-function-const>` check.

  Finds non-static member functions that can be made ``const``
  because the functions don't use ``this`` in a non-const way.

- New :doc:`readability-qualified-auto
  <clang-tidy/checks/readability-qualified-auto>` check.

  Adds pointer and ``const`` qualifications to ``auto``-typed variables
  that are deduced to pointers and ``const`` pointers.

- New :doc:`readability-redundant-access-specifiers
  <clang-tidy/checks/readability-redundant-access-specifiers>` check.

  Finds classes, structs, and unions that contain redundant member
  access specifiers.

New aliases
^^^^^^^^^^^

- New alias :doc:`cert-pos44-c
  <clang-tidy/checks/cert-pos44-c>` to
  :doc:`bugprone-bad-signal-to-kill-thread
  <clang-tidy/checks/bugprone-bad-signal-to-kill-thread>` was added.

- New alias :doc:`llvm-qualified-auto
  <clang-tidy/checks/llvm-qualified-auto>` to
  :doc:`readability-qualified-auto
  <clang-tidy/checks/readability-qualified-auto>` was added.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-posix-return
  <clang-tidy/checks/bugprone-posix-return>` check.

  Now also checks if any calls to ``pthread_*`` functions expect negative
  return values.

- Improved :doc:`hicpp-signed-bitwise
  <clang-tidy/checks/hicpp-signed-bitwise>` check.

  The check now supports the `IgnorePositiveIntegerLiterals` option.

- Improved :doc:`modernize-avoid-bind
  <clang-tidy/checks/modernize-avoid-bind>` check.

  The check now supports supports diagnosing and fixing arbitrary callables
  instead of only simple free functions. The `PermissiveParameterList` option
  has also been added to address situations where the existing fix-it logic
  would sometimes generate code that no longer compiles.

- The :doc:`modernize-use-equals-default
  <clang-tidy/checks/modernize-use-equals-default>` fix no longer adds
  semicolons where they would be redundant.

- Improved :doc:`modernize-use-override
  <clang-tidy/checks/modernize-use-override>` check.

  The check now supports the `AllowOverrideAndFinal` option to eliminate
  conflicts with `gcc -Wsuggest-override` or `gcc -Werror=suggest-override`.

- The :doc:`modernize-use-using
  <clang-tidy/checks/modernize-use-using>` check now converts typedefs
  containing struct definitions and multiple comma-separated types.

- Improved :doc:`readability-magic-numbers
  <clang-tidy/checks/readability-magic-numbers>` check.

  The check now supports the `IgnoreBitFieldsWidths` option to suppress
  the warning for numbers used to specify bit field widths.

  The check was updated to eliminate some false positives (such as using
  class enumeration as non-type template parameters, or the synthetically
  computed length of a static user string literal.)

- Improved :doc:`readability-redundant-member-init
  <clang-tidy/checks/readability-redundant-member-init>` check.

  The check  now supports the `IgnoreBaseInCopyConstructors` option to avoid
  `"base class 'Foo' should be explicitly initialized in the copy constructor"`
  warnings or errors with `gcc -Wextra` or `gcc -Werror=extra`.

- The :doc:`readability-redundant-string-init
  <clang-tidy/checks/readability-redundant-string-init>` check now supports a
  `StringNames` option enabling its application to custom string classes.

Renamed checks
^^^^^^^^^^^^^^

- The 'objc-avoid-spinlock' check was renamed to :doc:`darwin-avoid-spinlock
  <clang-tidy/checks/darwin-avoid-spinlock>`


Clang-tidy visual studio plugin
-------------------------------

The clang-tidy-vs plugin has been removed from clang, as
it's no longer maintained. Users should migrate to
`Clang Power Tools <https://marketplace.visualstudio.com/items?itemName=caphyon.ClangPowerTools>`_
instead.
