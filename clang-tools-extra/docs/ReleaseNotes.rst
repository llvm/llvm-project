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

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

New checks
^^^^^^^^^^

- New :doc:`readability-use-std-min-max
  <clang-tidy/checks/readability/use-std-min-max>` check.

  Replaces certain conditional statements with equivalent calls to
  ``std::min`` or ``std::max``.

New check aliases
^^^^^^^^^^^^^^^^^

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-suspicious-include
  <clang-tidy/checks/bugprone/suspicious-include>` check by replacing the local
  options `HeaderFileExtensions` and `ImplementationFileExtensions` by the
  global options of the same name.

- Cleaned up :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines/prefer-member-initializer>`
  by removing enforcement of rule `C.48
  <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c48-prefer-in-class-initializers-to-member-initializers-in-constructors-for-constant-initializers>`_,
  which was deprecated since :program:`clang-tidy` 17. This rule is now covered
  by :doc:`cppcoreguidelines-use-default-member-init
  <clang-tidy/checks/cppcoreguidelines/use-default-member-init>`.

- Improved :doc:`google-build-namespaces
  <clang-tidy/checks/google/build-namespaces>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.

- Improved :doc:`google-global-names-in-headers
  <clang-tidy/checks/google/global-names-in-headers>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.

- Improved :doc:`llvm-header-guard
  <clang-tidy/checks/llvm/header-guard>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.

- Improved :doc:`misc-definitions-in-headers
  <clang-tidy/checks/misc/definitions-in-headers>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.
  Additionally, the option `UseHeaderFileExtensions` is removed, so that the
  check uses the `HeaderFileExtensions` option unconditionally.

- Improved :doc:`misc-unused-using-decls
  <clang-tidy/checks/misc/unused-using-decls>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.

- Improved :doc:`misc-use-anonymous-namespace
  <clang-tidy/checks/misc/use-anonymous-namespace>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.

- Improved :doc:`modernize-avoid-c-arrays
  <clang-tidy/checks/modernize/avoid-c-arrays>` check by introducing the new
  `AllowStringArrays` option, enabling the exclusion of array types with deduced
  length initialized from string literals.

Removed checks
^^^^^^^^^^^^^^

Miscellaneous
^^^^^^^^^^^^^

- Removed `cert-dcl21-cpp`, which was deprecated since :program:`clang-tidy` 17,
  since the rule DCL21-CPP has been removed from the CERT guidelines.

- Fixed incorrect formatting in ``clang-apply-repalcements`` when no ``--format``
  option is specified. Now ``clang-apply-replacements`` applies formatting only with
  the option.

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
