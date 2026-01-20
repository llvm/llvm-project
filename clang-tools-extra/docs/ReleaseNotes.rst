.. If you want to modify sections/contents permanently, you should modify both
   ReleaseNotes.rst and ReleaseNotesTemplate.txt.

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

Potentially Breaking Changes
----------------------------

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

Improvements to clang-tidy
--------------------------

New checks
^^^^^^^^^^

New check aliases
^^^^^^^^^^^^^^^^^

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` check by adding the function
  ``std::get_temporary_buffer`` to the default list of unsafe functions. (This
  function is unsafe, useless, deprecated in C++17 and removed in C++20).

- Improved :doc:`misc-const-correctness
  <clang-tidy/checks/misc/const-correctness>` check:

  - Added support for analyzing function parameters with the `AnalyzeParameters`
    option.

- Improved :doc:`performance-move-const-arg
  <clang-tidy/checks/performance/move-const-arg>` check by avoiding false
  positives on trivially copyable types with a non-public copy constructor.

- Improved :doc:`readability-enum-initial-value
  <clang-tidy/checks/readability/enum-initial-value>` check: the warning message
  now uses separate note diagnostics for each uninitialized enumerator, making
  it easier to see which specific enumerators need explicit initialization.

Removed checks
^^^^^^^^^^^^^^

Miscellaneous
^^^^^^^^^^^^^

Improvements to include-fixer
-----------------------------

Improvements to clang-include-fixer
-----------------------------------

Improvements to modularize
--------------------------

Improvements to pp-trace
------------------------

Clang-tidy Visual Studio plugin
-------------------------------
