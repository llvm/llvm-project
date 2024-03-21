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

- Improved :program:`run-clang-tidy.py` script. Added argument `-source-filter`
  to filter source files from the compilation database, via a RegEx. In a
  similar fashion to what `-header-filter` does for header files.

New checks
^^^^^^^^^^

- New :doc:`bugprone-crtp-constructor-accessibility
  <clang-tidy/checks/bugprone/crtp-constructor-accessibility>` check.

  Detects error-prone Curiously Recurring Template Pattern usage, when the CRTP
  can be constructed outside itself and the derived class.

- New :doc:`bugprone-suspicious-stringview-data-usage
  <clang-tidy/checks/bugprone/suspicious-stringview-data-usage>` check.

  Identifies suspicious usages of ``std::string_view::data()`` that could lead
  to reading out-of-bounds data due to inadequate or incorrect string null
  termination.

- New :doc:`modernize-use-designated-initializers
  <clang-tidy/checks/modernize/use-designated-initializers>` check.

  Finds initializer lists for aggregate types that could be
  written as designated initializers instead.

- New :doc:`readability-use-std-min-max
  <clang-tidy/checks/readability/use-std-min-max>` check.

  Replaces certain conditional statements with equivalent calls to
  ``std::min`` or ``std::max``.

New check aliases
^^^^^^^^^^^^^^^^^

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-assert-side-effect
  <clang-tidy/checks/bugprone/assert-side-effect>` check by detecting side
  effect from calling a method with non-const reference parameters.

- Improved :doc:`bugprone-non-zero-enum-to-bool-conversion
  <clang-tidy/checks/bugprone/non-zero-enum-to-bool-conversion>` check by
  eliminating false positives resulting from direct usage of bitwise operators
  within parentheses.

- Improved :doc:`bugprone-suspicious-include
  <clang-tidy/checks/bugprone/suspicious-include>` check by replacing the local
  options `HeaderFileExtensions` and `ImplementationFileExtensions` by the
  global options of the same name.

- Improved :doc:`bugprone-too-small-loop-variable
  <clang-tidy/checks/bugprone/too-small-loop-variable>` check by incorporating
  better support for ``const`` loop boundaries.

- Improved :doc:`bugprone-unused-local-non-trivial-variable
  <clang-tidy/checks/bugprone/unused-local-non-trivial-variable>` check by
  ignoring local variable with ``[maybe_unused]`` attribute.

- Improved :doc:`bugprone-unused-return-value
  <clang-tidy/checks/bugprone/unused-return-value>` check by updating the
  parameter `CheckedFunctions` to support regexp, avoiding false positive for
  function with the same prefix as the default argument, e.g. ``std::unique_ptr``
  and ``std::unique``, avoiding false positive for assignment operator overloading.

- Improved :doc:`bugprone-use-after-move
  <clang-tidy/checks/bugprone/use-after-move>` check to also handle
  calls to ``std::forward``.

- Improved :doc:`cppcoreguidelines-missing-std-forward
  <clang-tidy/checks/cppcoreguidelines/missing-std-forward>` check by no longer
  giving false positives for deleted functions and fix false negative when some
  parameters are forwarded, but other aren't.

- Improved :doc:`cppcoreguidelines-owning-memory
  <clang-tidy/checks/cppcoreguidelines/owning-memory>` check to properly handle
  return type in lambdas and in nested functions.

- Cleaned up :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines/prefer-member-initializer>`
  by removing enforcement of rule `C.48
  <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c48-prefer-in-class-initializers-to-member-initializers-in-constructors-for-constant-initializers>`_,
  which was deprecated since :program:`clang-tidy` 17. This rule is now covered
  by :doc:`cppcoreguidelines-use-default-member-init
  <clang-tidy/checks/cppcoreguidelines/use-default-member-init>` and fixes
  incorrect hints when using list-initialization.

- Improved :doc:`google-build-namespaces
  <clang-tidy/checks/google/build-namespaces>` check by replacing the local
  option `HeaderFileExtensions` by the global option of the same name.

- Improved :doc:`google-explicit-constructor
  <clang-tidy/checks/google/explicit-constructor>` check to better handle
  ``C++-20`` `explicit(bool)`.

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

- Improved :doc:`modernize-loop-convert
  <clang-tidy/checks/modernize/loop-convert>` check by ensuring that fix-its
  don't remove parentheses used in ``sizeof`` calls when they have array index
  accesses as arguments.

- Improved :doc:`modernize-use-override
  <clang-tidy/checks/modernize/use-override>` check to also remove any trailing
  whitespace when deleting the ``virtual`` keyword.

- Improved :doc:`performance-unnecessary-copy-initialization
  <clang-tidy/checks/performance/unnecessary-copy-initialization>` check by
  detecting more cases of constant access. In particular, pointers can be
  analyzed, se the check now handles the common patterns
  `const auto e = (*vector_ptr)[i]` and `const auto e = vector_ptr->at(i);`.

- Improved :doc:`readability-implicit-bool-conversion
  <clang-tidy/checks/readability/implicit-bool-conversion>` check to provide
  valid fix suggestions for ``static_cast`` without a preceding space and
  fixed problem with duplicate parentheses in double implicit casts.

- Improved :doc:`readability-redundant-inline-specifier
  <clang-tidy/checks/readability/redundant-inline-specifier>` check to properly
  emit warnings for static data member with an in-class initializer.

- Improved :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check in `GetConfigPerFile`
  mode by resolving symbolic links to header files. Fixed handling of Hungarian
  Prefix when configured to `LowerCase`.

Removed checks
^^^^^^^^^^^^^^

- Removed `cert-dcl21-cpp`, which was deprecated since :program:`clang-tidy` 17,
  since the rule DCL21-CPP has been removed from the CERT guidelines.

Miscellaneous
^^^^^^^^^^^^^

- Fixed incorrect formatting in ``clang-apply-replacements`` when no ``--format``
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
