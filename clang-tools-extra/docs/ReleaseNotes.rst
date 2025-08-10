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

Code actions
^^^^^^^^^^^^

- New ``Override pure virtual methods`` code action. When invoked on a class
  definition, this action automatically generates C++ ``override`` declarations
  for all pure virtual methods inherited from its base classes that have not yet
  been implemented. The generated method stubs prompts the user for the actual
  implementation. The overrides are intelligently grouped under their original
  access specifiers (e.g., ``public``, ``protected``), creating new access
  specifier blocks if necessary.

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

Improvements to clang-query
---------------------------

- Matcher queries interpreted by clang-query are now support trailing comma (,)
  in matcher arguments. Note that C++ still doesn't allow this in function
  arguments. So when porting a query to C++, remove all instances of trailing
  comma (otherwise C++ compiler will just complain about "expected expression").

Improvements to clang-tidy
--------------------------

- The :program:`run-clang-tidy.py` and :program:`clang-tidy-diff.py` scripts
  now run checks in parallel by default using all available hardware threads.
  Both scripts display the number of threads being used in their output.

- Improved :program:`run-clang-tidy.py` by adding a new option
  `enable-check-profile` to enable per-check timing profiles and print a
  report based on all analyzed files.

New checks
^^^^^^^^^^

- New :doc:`bugprone-invalid-enum-default-initialization
  <clang-tidy/checks/bugprone/invalid-enum-default-initialization>` check.

  Detects default initialization (to 0) of variables with ``enum`` type where
  the enum has no enumerator with value of 0.

- New :doc:`llvm-mlir-op-builder
  <clang-tidy/checks/llvm/use-new-mlir-op-builder>` check.

  Checks for uses of MLIR's old/to be deprecated ``OpBuilder::create<T>`` form
  and suggests using ``T::create`` instead.

New check aliases
^^^^^^^^^^^^^^^^^

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-infinite-loop
  <clang-tidy/checks/bugprone/infinite-loop>` check by adding detection for
  variables introduced by structured bindings.

- Improved :doc:`bugprone-narrowing-conversions
  <clang-tidy/checks/bugprone/narrowing-conversions>` check by fixing
  false positive from analysis of a conditional expression in C.

- Improved :doc:`bugprone-reserved-identifier
  <clang-tidy/checks/bugprone/reserved-identifier>` check by ignoring
  declarations in system headers.

- Improved :doc:`bugprone-signed-char-misuse
  <clang-tidy/checks/bugprone/signed-char-misuse>` check by fixing
  false positives on C23 enums with the fixed underlying type of signed char.

- Improved :doc:`bugprone-tagged-union-member-count
  <clang-tidy/checks/bugprone/tagged-union-member-count>` by fixing a false
  positive when enums or unions from system header files or the ``std``
  namespace are treated as the tag or the data part of a user-defined
  tagged union respectively.

- Improved :doc:`bugprone-unhandled-self-assignment
  <clang-tidy/checks/bugprone/unhandled-self-assignment>` check by adding
  an additional matcher that generalizes the copy-and-swap idiom pattern
  detection.

- Improved :doc:`misc-header-include-cycle
  <clang-tidy/checks/misc/header-include-cycle>` check performance.

- Improved :doc:`modernize-use-designated-initializers
  <clang-tidy/checks/modernize/use-designated-initializers>` check to
  suggest using designated initializers for aliased aggregate types.

- Improved :doc:`modernize-use-std-format
  <clang-tidy/checks/modernize/use-std-format>` check to correctly match
  when the format string is converted to a different type by an implicit
  constructor call.

- Improved :doc:`modernize-use-std-print
  <clang-tidy/checks/modernize/use-std-print>` check to correctly match
  when the format string is converted to a different type by an implicit
  constructor call.

- Improved :doc:`performance-unnecessary-copy-initialization
  <clang-tidy/checks/performance/unnecessary-copy-initialization>` by printing
  the type of the diagnosed variable.

- Improved :doc:`performance-unnecessary-value-param
  <clang-tidy/checks/performance/unnecessary-value-param>` by printing
  the type of the diagnosed variable.

- Improved :doc:`portability-template-virtual-member-function
  <clang-tidy/checks/portability/template-virtual-member-function>` check to
  avoid false positives on pure virtual member functions.

- Improved :doc:`readability-container-size-empty
  <clang-tidy/checks/readability/container-size-empty>` check by correctly
  generating fix-it hints when size method is called from implicit ``this``.

- Improved :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check by ignoring
  declarations in system headers.

- Improved :doc:`readability-qualified-auto
  <clang-tidy/checks/readability/qualified-auto>` check by adding the option
  `IgnoreAliasing`, that allows not looking at underlying types of type aliases.

Removed checks
^^^^^^^^^^^^^^

Miscellaneous
^^^^^^^^^^^^^

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
