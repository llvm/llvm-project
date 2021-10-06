====================================================
Extra Clang Tools 13.0.0 (In-Progress) Release Notes
====================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 13 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 13.0.0. Here we describe the status of the Extra Clang Tools in
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

What's New in Extra Clang Tools 13.0.0?
=======================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

...

Improvements to clangd
----------------------

Code Completion
^^^^^^^^^^^^^^^

- ML based model is used by default for ranking completion candidates.

- Support for completion of attributes.

- Improved handling of Objective-C(++) constructs.


Hover
^^^^^

- Shows documentation for Attributes.

- Displays resolved paths for includes.

- Shows padding for fields.

Document Outline
^^^^^^^^^^^^^^^^

- Contains information in detail field about extra type information

- Macro expansions now show up in the tree

- Improved handling of Objective-C(++) constructs.

Code Navigation
^^^^^^^^^^^^^^^^

- Cross references surfaces occurrences for calls to overridden methods and
  declarations.

Semantic Highlighting
^^^^^^^^^^^^^^^^^^^^^

- Support for legacy semantic tokens extension is dropped.

- Better support for Objective-C(++) constructs and dependent code.


Diagnostics
^^^^^^^^^^^

- Diagnostics for unused/deprecated code are tagged according to LSP.

- Clang-tidy checks that operate at translation-unit level are now available.

System Integration
^^^^^^^^^^^^^^^^^^

- Compile flag parsing has been improved to be more resilient against multiple
  jobs.

- Better error reporting when compile flags are unusable.


Miscellaneous
^^^^^^^^^^^^^

- Better support for TUs with circular includes (e.g. templated header vs
  implementation file).

- Compile flags for headers are inferred from files known to be including them
  when possible.

- Version info contains information about compile-time setup of clangd

- FeatureModule mechanism has been introduced to make contribution of vertical
  features to clangd easier, by making it possible to write features that can
  interact with clangd-core without touching it.

- There's an extension for inlay-hints for deduced types and parameter names,
  hidden behind -inlay-hints flag.

- Rename is more robust:

  - Won't trigger on non-identifiers.
  - Makes use of dirty buffers for open files.

- Improvements to dex query latency.

- There's a remote-index service for LLVM at http://clangd-index.llvm.org/.

- There's a remote-index service for Chromium at
  https://linux.clangd-index.chromium.org/.

Improvements to clang-doc
-------------------------

The improvements are...

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- The `run-clang-tidy.py` helper script is now installed in `bin/` as
  `run-clang-tidy`. It was previously installed in `share/clang/`.

- Added command line option `--fix-notes` to apply fixes found in notes
  attached to warnings. These are typically cases where we are less confident
  the fix will have the desired effect.

- libToolingCore and Clang-Tidy was refactored and now checks can produce
  highlights (`^~~~~` under fragments of the source code) in diagnostics.
  Existing and new checks in the future can be expected to start implementing
  this functionality.
  This change only affects the visual rendering of diagnostics, and does not
  alter the behavior of generated fixes.

New checks
^^^^^^^^^^


- New :doc:`altera-id-dependent-backward-branch
  <clang-tidy/checks/altera-id-dependent-backward-branch>` check.

  Finds ID-dependent variables and fields that are used within loops. This
  causes branches to occur inside the loops, and thus leads to performance
  degradation.

- New :doc:`altera-unroll-loops
  <clang-tidy/checks/altera-unroll-loops>` check.

  Finds inner loops that have not been unrolled, as well as fully unrolled
  loops with unknown loops bounds or a large number of iterations.

- New :doc:`bugprone-easily-swappable-parameters
  <clang-tidy/checks/bugprone-easily-swappable-parameters>` check.

  Finds function definitions where parameters of convertible types follow each
  other directly, making call sites prone to calling the function with
  swapped (or badly ordered) arguments.

- New :doc:`bugprone-implicit-widening-of-multiplication-result
  <clang-tidy/checks/bugprone-implicit-widening-of-multiplication-result>` check.

  Diagnoses instances of an implicit widening of multiplication result.

- New :doc:`bugprone-unhandled-exception-at-new
  <clang-tidy/checks/bugprone-unhandled-exception-at-new>` check.

  Finds calls to ``new`` with missing exception handler for ``std::bad_alloc``.

- New :doc:`concurrency-thread-canceltype-asynchronous
  <clang-tidy/checks/concurrency-thread-canceltype-asynchronous>` check.

  Finds ``pthread_setcanceltype`` function calls where a thread's cancellation
  type is set to asynchronous.

- New :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines-prefer-member-initializer>` check.

  Finds member initializations in the constructor body which can be placed into
  the initialization list instead.

- New :doc:`readability-suspicious-call-argument
  <clang-tidy/checks/readability-suspicious-call-argument>` check.

  Finds function calls where the arguments passed are provided out of order,
  based on the difference between the argument name and the parameter names
  of the function.

New check aliases
^^^^^^^^^^^^^^^^^

- New alias :doc:`cert-pos47-c
  <clang-tidy/checks/cert-pos47-c>` to
  :doc:`concurrency-thread-canceltype-asynchronous
  <clang-tidy/checks/concurrency-thread-canceltype-asynchronous>` was added.


Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-signal-handler
  <clang-tidy/checks/bugprone-signal-handler>` check.

  Added an option to choose the set of allowed functions.

- Improved :doc:`cppcoreguidelines-init-variables
  <clang-tidy/checks/cppcoreguidelines-init-variables>` check.

  Removed generating fixes for enums because the code generated was broken,
  trying to initialize the enum from an integer.

  The check now also warns for uninitialized scoped enums.

- Improved :doc:`readability-uniqueptr-delete-release
  <clang-tidy/checks/readability-uniqueptr-delete-release>` check.

  Added an option to choose whether to refactor by calling the ``reset`` member
  function or assignment to ``nullptr``.
  Added support for pointers to ``std::unique_ptr``.


Removed checks
^^^^^^^^^^^^^^

- The readability-deleted-default check has been removed.

  The clang warning `Wdefaulted-function-deleted
  <https://clang.llvm.org/docs/DiagnosticsReference.html#wdefaulted-function-deleted>`_
  will diagnose the same issues and is enabled by default.

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
