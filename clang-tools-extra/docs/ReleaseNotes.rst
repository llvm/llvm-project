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

The improvements are...

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

New checks
^^^^^^^^^^

- New :doc:`concurrency-thread-canceltype-asynchronous
  <clang-tidy/checks/concurrency-thread-canceltype-asynchronous>` check.

  Finds ``pthread_setcanceltype`` function calls where a thread's cancellation
  type is set to asynchronous.

- New :doc:`altera-unroll-loops
  <clang-tidy/checks/altera-unroll-loops>` check.

  Finds inner loops that have not been unrolled, as well as fully unrolled
  loops with unknown loops bounds or a large number of iterations.

- New :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines-prefer-member-initializer>` check.

  Finds member initializations in the constructor body which can be placed into
  the initialization list instead.

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
