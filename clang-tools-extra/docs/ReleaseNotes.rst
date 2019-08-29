=====================================
Extra Clang Tools 9.0.0 Release Notes
=====================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 9.0.0. Here we describe the status of the Extra Clang Tools in
some detail, including major improvements from the previous release and new
feature work. All LLVM releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or
the `LLVM Web Site <https://llvm.org>`_.


What's New in Extra Clang Tools 9.0.0?
======================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.


Improvements to clangd
----------------------

- Background indexing is on by default

  When using clangd, it will build an index of your code base (all files listed
  in your compile database). This index enables go-to-definition,
  find-references, and even code completion to find symbols across your project.

  This feature can consume a lot of CPU. It can be disabled using the
  ``--background-index=false`` flag, and respects ``-j`` to use fewer threads.
  The index is written to ``.clangd/index`` in the project root.

- Contextual code actions

  Extract variable, expand ``auto``, expand macro, convert string to raw string.
  More to come in the future!

- Clang-tidy warnings are available

  These will be produced for projects that have a ``.clang-tidy`` file in their
  source tree, as described in the :doc:`clang-tidy documentation <clang-tidy>`.

- Improved diagnostics

  Errors from headers are now shown (on the #including line).
  The message now indicates if fixes are available.
  Navigation between errors and associated notes is improved (for editors that
  support ``Diagnostic.relatedInformation``).

- Suggested includes

  When a class or other name is not found, clangd may suggest to fix this by
  adding the corresponding ``#include`` directive.

- Semantic highlighting

  clangd can push syntax information to the editor, allowing it to highlight
  e.g. member variables differently from locals. (requires editor support)

  This implements the proposed protocol from
  https://github.com/microsoft/vscode-languageserver-node/pull/367

- Type hierachy

  Navigation to base/derived types is possible in editors that support the
  proposed protocol from
  https://github.com/microsoft/vscode-languageserver-node/pull/426

- Improvements to include insertion

  Only headers with ``#include``-guards will be inserted, and the feature can
  be disabled with the ``--header-insertion=never`` flag.

  Standard library headers should now be inserted more accurately, particularly
  for C++ other than libstdc++, and for the C standard library.

- Code completion

  Overloads are bundled into a single completion item by default. (for editors
  that support signature-help).

  Redundant const/non-const overloads are no longer shown.

  Before clangd is warmed up (during preamble build), limited identifier- and
  index-based code completion is available.

- Format-on-type

  A new implementation of format-on-type is triggered by hitting enter: it
  attempts to reformat the previous line and reindent the new line.
  (Requires editor support).

- Toolchain header detection

  Projects that use an embedded gcc toolchain may only work when used with the
  corresponding standard library. clangd can now query the toolchain to find
  these headers.
  The compilation database must correctly specify this toolchain, and the
  ``--query-driver=/path/to/toolchain/bin/*`` flag must be passed to clangd.

- Miscellaneous improvements

  Hover now produces richer Markdown-formatted text (for supported editors).

  Rename is safer and more helpful, though is still within one file only.

  Files without extensions (e.g. C++ standard library) are handled better.

  clangd can understand offsets in UTF-8 or UTF-32 through command-line flags or
  protocol extensions. (Useful with editors/platforms that don't speak UTF-16).

  Editors that support edits near the cursor in code-completion can set the
  ``textDocument.completion.editsNearCursor`` capability to ``true``, and clangd
  will provide completions that correct ``.`` to ``->``, and vice-versa.


Improvements to clang-tidy
--------------------------

- New OpenMP module.

  For checks specific to `OpenMP <https://www.openmp.org/>`_ API.

- New :doc:`abseil-duration-addition
  <clang-tidy/checks/abseil-duration-addition>` check.

  Checks for cases where addition should be performed in the ``absl::Time``
  domain.

- New :doc:`abseil-duration-conversion-cast
  <clang-tidy/checks/abseil-duration-conversion-cast>` check.

  Checks for casts of ``absl::Duration`` conversion functions, and recommends
  the right conversion function instead.

- New :doc:`abseil-duration-unnecessary-conversion
  <clang-tidy/checks/abseil-duration-unnecessary-conversion>` check.

  Finds and fixes cases where ``absl::Duration`` values are being converted to
  numeric types and back again.

- New :doc:`abseil-time-comparison
  <clang-tidy/checks/abseil-time-comparison>` check.

  Prefer comparisons in the ``absl::Time`` domain instead of the integer
  domain.

- New :doc:`abseil-time-subtraction
  <clang-tidy/checks/abseil-time-subtraction>` check.

  Finds and fixes ``absl::Time`` subtraction expressions to do subtraction
  in the Time domain instead of the numeric domain.

- New :doc:`android-cloexec-pipe
  <clang-tidy/checks/android-cloexec-pipe>` check.

  This check detects usage of ``pipe()``.

- New :doc:`android-cloexec-pipe2
  <clang-tidy/checks/android-cloexec-pipe2>` check.

  This checks ensures that ``pipe2()`` is called with the ``O_CLOEXEC`` flag.

- New :doc:`bugprone-branch-clone
  <clang-tidy/checks/bugprone-branch-clone>` check.

  Checks for repeated branches in ``if/else if/else`` chains, consecutive
  repeated branches in ``switch`` statements and indentical true and false
  branches in conditional operators.

- New :doc:`bugprone-posix-return
  <clang-tidy/checks/bugprone-posix-return>` check.

  Checks if any calls to POSIX functions (except ``posix_openpt``) expect negative
  return values.

- New :doc:`bugprone-unhandled-self-assignment
  <clang-tidy/checks/bugprone-unhandled-self-assignment>` check.

  Finds user-defined copy assignment operators which do not protect the code
  against self-assignment either by checking self-assignment explicitly or
  using the copy-and-swap or the copy-and-move method.

- New :doc:`fuchsia-default-arguments-calls
  <clang-tidy/checks/fuchsia-default-arguments-calls>` check.

  Warns if a function or method is called with default arguments.
  This was previously done by `fuchsia-default-arguments check`, which has been
  removed.

- New :doc:`fuchsia-default-arguments-declarations
  <clang-tidy/checks/fuchsia-default-arguments-declarations>` check.

  Warns if a function or method is declared with default parameters.
  This was previously done by `fuchsia-default-arguments check` check, which has
  been removed.

- New :doc:`google-objc-avoid-nsobject-new
  <clang-tidy/checks/google-objc-avoid-nsobject-new>` check.

  Checks for calls to ``+new`` or overrides of it, which are prohibited by the
  Google Objective-C style guide.

- New :doc:`google-readability-avoid-underscore-in-googletest-name
  <clang-tidy/checks/google-readability-avoid-underscore-in-googletest-name>`
  check.

  Checks whether there are underscores in googletest test and test case names in
  test macros, which is prohibited by the Googletest FAQ.

- New :doc:`llvm-prefer-isa-or-dyn-cast-in-conditionals
  <clang-tidy/checks/llvm-prefer-isa-or-dyn-cast-in-conditionals>` check.

  Looks at conditionals and finds and replaces cases of ``cast<>``,
  which will assert rather than return a null pointer, and
  ``dyn_cast<>`` where the return value is not captured. Additionally,
  finds and replaces cases that match the pattern ``var &&
  isa<X>(var)``, where ``var`` is evaluated twice.

- New :doc:`modernize-use-trailing-return-type
  <clang-tidy/checks/modernize-use-trailing-return-type>` check.

  Rewrites function signatures to use a trailing return type.

- New :doc:`objc-super-self <clang-tidy/checks/objc-super-self>` check.

  Finds invocations of ``-self`` on super instances in initializers of
  subclasses of ``NSObject`` and recommends calling a superclass initializer
  instead.

- New :doc:`openmp-exception-escape
  <clang-tidy/checks/openmp-exception-escape>` check.

  Analyzes OpenMP Structured Blocks and checks that no exception escapes
  out of the Structured Block it was thrown in.

- New :doc:`openmp-use-default-none
  <clang-tidy/checks/openmp-use-default-none>` check.

  Finds OpenMP directives that are allowed to contain a ``default`` clause,
  but either don't specify it or the clause is specified but with the kind
  other than ``none``, and suggests to use the ``default(none)`` clause.

- New :doc:`readability-convert-member-functions-to-static
  <clang-tidy/checks/readability-convert-member-functions-to-static>` check.

  Finds non-static member functions that can be made ``static``.

- New alias :doc:`cert-oop54-cpp
  <clang-tidy/checks/cert-oop54-cpp>` to
  :doc:`bugprone-unhandled-self-assignment
  <clang-tidy/checks/bugprone-unhandled-self-assignment>` was added.

- New alias :doc:`cppcoreguidelines-explicit-virtual-functions
  <clang-tidy/checks/cppcoreguidelines-explicit-virtual-functions>` to
  :doc:`modernize-use-override
  <clang-tidy/checks/modernize-use-override>` was added.

- The :doc:`bugprone-argument-comment
  <clang-tidy/checks/bugprone-argument-comment>` now supports
  `CommentBoolLiterals`, `CommentIntegerLiterals`, `CommentFloatLiterals`,
  `CommentUserDefiniedLiterals`, `CommentStringLiterals`,
  `CommentCharacterLiterals` & `CommentNullPtrs` options.

- The :doc:`bugprone-too-small-loop-variable
  <clang-tidy/checks/bugprone-too-small-loop-variable>` now supports
  `MagnitudeBitsUpperLimit` option. The default value was set to 16,
  which greatly reduces warnings related to loops which are unlikely to
  cause an actual functional bug.

- Added `UseAssignment` option to :doc:`cppcoreguidelines-pro-type-member-init
  <clang-tidy/checks/cppcoreguidelines-pro-type-member-init>`

  If set to true, the check will provide fix-its with literal initializers
  (``int i = 0;``) instead of curly braces (``int i{};``).

- The `fuchsia-default-arguments` check has been removed.

  Warnings of function or method calls and declarations with default arguments
  were moved to :doc:`fuchsia-default-arguments-calls
  <clang-tidy/checks/fuchsia-default-arguments-calls>` and
  :doc:`fuchsia-default-arguments-declarations
  <clang-tidy/checks/fuchsia-default-arguments-declarations>` checks
  respectively.

- The :doc:`google-runtime-int <clang-tidy/checks/google-runtime-int>`
  check has been disabled in Objective-C++.

- The :doc:`modernize-use-override
  <clang-tidy/checks/modernize-use-override>` now supports `OverrideSpelling`
  and `FinalSpelling` options.

- The :doc:`misc-throw-by-value-catch-by-reference
  <clang-tidy/checks/misc-throw-by-value-catch-by-reference>` now supports
  `WarnOnLargeObject` and `MaxSize` options to warn on any large trivial
  object caught by value.

- The `Acronyms` and `IncludeDefaultAcronyms` options for the
  :doc:`objc-property-declaration <clang-tidy/checks/objc-property-declaration>`
  check have been removed.


Improvements to pp-trace
------------------------

- Added a new option `-callbacks` to filter preprocessor callbacks. It replaces
  the `-ignore` option.
