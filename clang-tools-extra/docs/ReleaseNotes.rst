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

- New :doc:`bugprone-dynamic-static-initializers
  <clang-tidy/checks/bugprone-dynamic-static-initializers>` check.

  Finds instances where variables with static storage are initialized
  dynamically in header files.

- New :doc:`linuxkernel-must-use-errs
  <clang-tidy/checks/linuxkernel-must-use-errs>` check.

  Checks Linux kernel code to see if it uses the results from the functions in
  ``linux/err.h``.

- New :doc:`google-upgrade-googletest-case
  <clang-tidy/checks/google-upgrade-googletest-case>` check.

  Finds uses of deprecated Googletest APIs with names containing ``case`` and
  replaces them with equivalent APIs with ``suite``.

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
