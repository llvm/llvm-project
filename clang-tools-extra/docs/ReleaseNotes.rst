====================================================
Extra Clang Tools |release| |ReleaseNotesTitle|
====================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming Extra Clang Tools |version| release.
     Release notes for previous releases can be found on
     `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release |release|. Here we describe the status of the Extra Clang Tools in
some detail, including major improvements from the previous release and new
feature work. All LLVM releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or
the `LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Extra Clang Tools |release|?
==========================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

...

Improvements to clangd
----------------------

Inlay hints
^^^^^^^^^^^

Diagnostics
^^^^^^^^^^^

Semantic Highlighting
^^^^^^^^^^^^^^^^^^^^^

Compile flags
^^^^^^^^^^^^^

Hover
^^^^^

Code completion
^^^^^^^^^^^^^^^

Signature help
^^^^^^^^^^^^^^

Cross-references
^^^^^^^^^^^^^^^^

Objective-C
^^^^^^^^^^^

Miscellaneous
^^^^^^^^^^^^^

Improvements to clang-doc
-------------------------

- The default executor was changed to standalone to match other tools.

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

New checks
^^^^^^^^^^

- New :doc:`bugprone-suspicious-realloc-usage
  <clang-tidy/checks/bugprone/suspicious-realloc-usage>` check.

  Finds usages of ``realloc`` where the return value is assigned to the
  same expression as passed to the first argument.

- New :doc:`cppcoreguidelines-avoid-const-or-ref-data-members
  <clang-tidy/checks/cppcoreguidelines/avoid-const-or-ref-data-members>` check.

  Warns when a struct or class uses const or reference (lvalue or rvalue) data members.

- New :doc:`cppcoreguidelines-avoid-do-while
  <clang-tidy/checks/cppcoreguidelines/avoid-do-while>` check.

  Warns when using ``do-while`` loops.

New check aliases
^^^^^^^^^^^^^^^^^

- New alias :doc:`cert-msc54-cpp
  <clang-tidy/checks/cert/msc54-cpp>` to
  :doc:`bugprone-signal-handler
  <clang-tidy/checks/bugprone/signal-handler>` was added.


Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a false positive in :doc:`bugprone-assignment-in-if-condition
  <clang-tidy/checks/bugprone/assignment-in-if-condition>` check when there
  was an assignement in a lambda found in the condition of an ``if``.

- Improved :doc:`bugprone-signal-handler
  <clang-tidy/checks/bugprone/signal-handler>` check. Partial
  support for C++14 signal handler rules was added. Bug report generation was
  improved.

- Fixed a false positive in :doc:`cppcoreguidelines-pro-type-member-init
  <clang-tidy/checks/cppcoreguidelines/pro-type-member-init>` when warnings
  would be emitted for uninitialized members of an anonymous union despite
  there being an initializer for one of the other members.

- Improved :doc:`modernize-use-emplace <clang-tidy/checks/modernize/use-emplace>`
  check.

  The check now supports detecting inefficient invocations of ``push`` and
  ``push_front`` on STL-style containers and replacing them with ``emplace``
  or ``emplace_front``.

  The check now supports detecting alias cases of ``push_back`` ``push`` and
  ``push_front`` on STL-style containers and replacing them with ``emplace_back``,
  ``emplace`` or ``emplace_front``.

- Improved :doc:`modernize-use-equals-default <clang-tidy/checks/modernize/use-equals-default>`
  check.

  The check now skips unions/union-like classes since in this case a default constructor
  with empty body is not equivalent to the explicitly defaulted one, variadic constructors
  since they cannot be explicitly defaulted. The check also skips copy assignment operators
  with nonstandard return types, private/protected default constructors for C++17 or earlier.
  The check is restricted to C++11 or later.

- Change the default behavior of :doc:`readability-avoid-const-params-in-decls
  <clang-tidy/checks/readability/avoid-const-params-in-decls>` to not
  warn about `const` value parameters of declarations inside macros.

- Fixed crashes in :doc:`readability-braces-around-statements
  <clang-tidy/checks/readability/braces-around-statements>` and
  :doc:`readability-simplify-boolean-expr <clang-tidy/checks/readability/simplify-boolean-expr>`
  when using a C++23 ``if consteval`` statement.

- Improved :doc:`misc-redundant-expression <clang-tidy/checks/misc/redundant-expression>`
  check.

  The check now skips concept definitions since redundant expressions still make sense
  inside them.

Removed checks
^^^^^^^^^^^^^^

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

Clang-tidy Visual Studio plugin
-------------------------------
