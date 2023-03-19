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

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- New global configuration file options `HeaderFileExtensions` and
  `ImplementationFileExtensions`, replacing the check-local options of the
  same name.

New checks
^^^^^^^^^^

- New :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` check.

  Checks for functions that have safer, more secure replacements available, or
  are considered deprecated due to design flaws.
  This check relies heavily on, but is not exclusive to, the functions from
  the *Annex K. "Bounds-checking interfaces"* of C11.

- New :doc:`cppcoreguidelines-avoid-capture-default-when-capturing-this
  <clang-tidy/checks/cppcoreguidelines/avoid-capture-default-when-capturing-this>` check.

  Warns when lambda specify a capture default and capture ``this``.

- New :doc:`cppcoreguidelines-avoid-capturing-lambda-coroutines
  <clang-tidy/checks/cppcoreguidelines/avoid-capturing-lambda-coroutines>` check.

  Flags C++20 coroutine lambdas with non-empty capture lists that may cause
  use-after-free errors and suggests avoiding captures or ensuring the lambda
  closure object has a guaranteed lifetime.

- New :doc:`cppcoreguidelines-rvalue-reference-param-not-moved
  <clang-tidy/checks/cppcoreguidelines/rvalue-reference-param-not-moved>` check.

  Warns when an rvalue reference function parameter is never moved within
  the function body.

- New :doc:`llvmlibc-inline-function-decl
  <clang-tidy/checks/llvmlibc/inline-function-decl>` check.

  Checks that all implicit and explicit inline functions in header files are
  tagged with the ``LIBC_INLINE`` macro.

New check aliases
^^^^^^^^^^^^^^^^^

- New alias :doc:`cert-msc24-c
  <clang-tidy/checks/cert/msc24-c>` to :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` was added.

- New alias :doc:`cert-msc33-c
  <clang-tidy/checks/cert/msc33-c>` to :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` was added.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Improved :doc:`readability-redundant-string-cstr
  <clang-tidy/checks/readability/redundant-string-cstr>` check to recognise
  unnecessary ``std::string::c_str()`` and ``std::string::data()`` calls in
  arguments to ``std::print``, ``std::format`` or other functions listed in
  the ``StringParameterFunction`` check option.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`bugprone-dynamic-static-initializers
  <clang-tidy/checks/bugprone/dynamic-static-initializers>` check.
  Global options of the same name should be used instead.

- Deprecated check-local options `HeaderFileExtensions` and `ImplementationFileExtensions`
  in :doc:`bugprone-suspicious-include
  <clang-tidy/checks/bugprone/suspicious-include>` check.
  Global options of the same name should be used instead.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`google-build-namespaces
  <clang-tidy/checks/google/build-namespaces>` check.
  Global options of the same name should be used instead.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`google-global-names-in-headers
  <clang-tidy/checks/google/global-names-in-headers>` check.
  Global options of the same name should be used instead.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`llvm-header-guard
  <clang-tidy/checks/llvm/header-guard>` check.
  Global options of the same name should be used instead.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`misc-definitions-in-headers
  <clang-tidy/checks/misc/definitions-in-headers>` check.
  Global options of the same name should be used instead.

- Deprecated check-local options `HeaderFileExtensions`
  in :doc:`misc-unused-using-decls
  <clang-tidy/checks/misc/unused-using-decls>` check.
  Global options of the same name should be used instead.

- In :doc:`modernize-use-default-member-init
  <clang-tidy/checks/modernize/use-default-member-init>` count template
  constructors toward hand written constructors so that they are skipped if more
  than one exists.

- Fixed reading `HungarianNotation.CString.*` options in
  :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check.

- Renamed `HungarianNotation.CString` options `CharPrinter` and
  `WideCharPrinter` to `CharPointer` and `WideCharPointer` respectively in
  :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check.

- Updated the Hungarian prefixes for enums in C files to match those used in C++
  files for improved readability, as checked by :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>`. To preserve the previous
  behavior of using `i` as the prefix for enum tags, set the `EnumConstantPrefix`
  option to `i` instead of using `EnumConstantHungarianPrefix`.

- Fixed a false positive in :doc:`readability-container-size-empty
  <clang-tidy/checks/readability/container-size-empty>` check when comparing
  ``std::array`` objects to default constructed ones. The behavior for this and
  other relevant classes can now be configured with a new option.

- Improved :doc:`bugprone-too-small-loop-variable
  <clang-tidy/checks/bugprone/too-small-loop-variable>` check. Basic support
  for bit-field and integer members as a loop variable or upper limit were added.

- Improved :doc:`readability-magic-numbers
  <clang-tidy/checks/readability/magic-numbers>` check, now allows for
  magic numbers in type aliases such as ``using`` and ``typedef`` declarations if
  the new ``IgnoreTypeAliases`` option is set to true.

- Fixed a false positive in :doc:`cppcoreguidelines-slicing
  <clang-tidy/checks/cppcoreguidelines/slicing>` check when warning would be
  emitted in constructor for virtual base class initialization.

- Improved :doc:`bugprone-use-after-move
  <clang-tidy/checks/bugprone/use-after-move>` to understand that there is a
  sequence point between designated initializers.

- Fixed an issue in :doc:`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` when specifying an empty
  string for ``Prefix`` or ``Suffix`` options could result in the style not
  being used.

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
