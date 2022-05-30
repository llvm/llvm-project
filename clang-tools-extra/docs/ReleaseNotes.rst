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
- Improved Fix-its of some clang-tidy checks when applied with clangd.

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

The improvements are...

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- Added trace code to help narrow down any checks and the relevant source code
  that result in crashes.

- Clang-tidy now consideres newlines as separators of single elements in the `Checks` section in
  `.clang-tidy` configuration files. Where previously a comma had to be used to distinguish elements in
  this list from each other, newline characters now also work as separators in the parsed YAML. That
  means it is advised to use YAML's block style initiated by the pipe character `|` for the `Checks`
  section in order to benefit from the easier syntax that works without commas.

- Fixed a regression introduced in clang-tidy 14.0.0, which prevented NOLINTs
  from suppressing diagnostics associated with macro arguments. This fixes
  `Issue 55134 <https://github.com/llvm/llvm-project/issues/55134>`_.

New checks
^^^^^^^^^^

- New :doc:`bugprone-shared-ptr-array-mismatch <clang-tidy/checks/bugprone-shared-ptr-array-mismatch>` check.

  Finds initializations of C++ shared pointers to non-array type that are initialized with an array.

- New :doc:`bugprone-unchecked-optional-access
  <clang-tidy/checks/bugprone-unchecked-optional-access>` check.

   Warns when the code is unwrapping a `std::optional<T>`, `absl::optional<T>`,
   or `base::Optional<T>` object without assuring that it contains a value.

- New :doc:`modernize-macro-to-enum
  <clang-tidy/checks/modernize-macro-to-enum>` check.

  Replaces groups of adjacent macros with an unscoped anonymous enum.

- New :doc:`portability-std-allocator-const <clang-tidy/checks/portability-std-allocator-const>` check.

  Report use of ``std::vector<const T>`` (and similar containers of const
  elements). These are not allowed in standard C++ due to undefined
  ``std::allocator<const T>``. They do not compile with libstdc++ or MSVC.
  Future libc++ will remove the extension (`D120996
  <https://reviews.llvm.org/D120996>`).

New check aliases
^^^^^^^^^^^^^^^^^

- New alias :doc:`cppcoreguidelines-macro-to-enum
  <clang-tidy/checks/cppcoreguidelines-macro-to-enum>` to :doc:`modernize-macro-to-enum
  <clang-tidy/checks/modernize-macro-to-enum>` was added.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed nonsensical suggestion of :doc:`altera-struct-pack-align
  <clang-tidy/checks/altera-struct-pack-align>` check for empty structs.

- Fixed some false positives in :doc:`bugprone-infinite-loop
  <clang-tidy/checks/bugprone-infinite-loop>` involving dependent expressions.

- Fixed a crash in :doc:`bugprone-sizeof-expression
  <clang-tidy/checks/bugprone-sizeof-expression>` when `sizeof(...)` is
  compared against a `__int128_t`.

- Improved :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines-prefer-member-initializer>` check.

  Fixed an issue when there was already an initializer in the constructor and
  the check would try to create another initializer for the same member.

- Fixed a crash in :doc:`llvmlibc-callee-namespace
  <clang-tidy/checks/llvmlibc-callee-namespace>` when executing for C++ code
  that contain calls to advanced constructs, e.g. overloaded operators.

- Fixed a false positive in :doc:`misc-redundant-expression
  <clang-tidy/checks/misc-redundant-expression>` involving overloaded
  comparison operators.

- Fixed a false positive in :doc:`misc-redundant-expression
  <clang-tidy/checks/misc-redundant-expression>` involving assignments in
  conditions. This fixes `Issue 35853 <https://github.com/llvm/llvm-project/issues/35853>`_.

- Fixed a false positive in :doc:`modernize-deprecated-headers
  <clang-tidy/checks/modernize-deprecated-headers>` involving including
  C header files from C++ files wrapped by ``extern "C" { ... }`` blocks.
  Such includes will be ignored by now.
  By default now it doesn't warn for including deprecated headers from header
  files, since that header file might be used from C source files. By passing
  the ``CheckHeaderFile=true`` option if header files of the project only
  included by C++ source files.

- Improved :doc:`performance-inefficient-vector-operation
  <clang-tidy/checks/performance-inefficient-vector-operation>` to work when
  the vector is a member of a structure.

- Fixed a crash in :doc:`readability-const-return-type
  <clang-tidy/checks/readability-const-return-type>` when a pure virtual function
  overrided has a const return type. Removed the fix for a virtual function.

- Fixed incorrect suggestions for :doc:`readability-container-size-empty
  <clang-tidy/checks/readability-container-size-empty>` when smart pointers are involved.

- Fixed a false positive in :doc:`readability-non-const-parameter
  <clang-tidy/checks/readability-non-const-parameter>` when the parameter is
  referenced by an lvalue.

- Expanded :doc:`readability-simplify-boolean-expr
  <clang-tidy/checks/readability-simplify-boolean-expr>` to simplify expressions
  using DeMorgan's Theorem.

- Fixed a crash in :doc:`performance-unnecessary-value-param
  <clang-tidy/checks/readability-suspicious-call-argument>` when the specialization
  template has an unnecessary value paramter. Removed the fix for a template.

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

The improvements are...

Clang-tidy Visual Studio plugin
-------------------------------
