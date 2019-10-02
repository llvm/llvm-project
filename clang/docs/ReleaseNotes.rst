========================================
Clang 10.0.0 (In-Progress) Release Notes
========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 10 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 10.0.0. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Subversion checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Clang 10.0.0?
===========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- ...

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- -Wtautological-overlap-compare will warn on negative numbers and non-int
  types.
- -Wtautological-compare for self comparisons and
  -Wtautological-overlap-compare will now look through member and array
  access to determine if two operand expressions are the same.

Non-comprehensive list of changes in this release
-------------------------------------------------

- For X86 target, -march=skylake-avx512, -march=icelake-client,
  -march=icelake-server, -march=cascadelake, -march=cooperlake will default to
  not using 512-bit zmm registers in vectorized code unless 512-bit intrinsics
  are used in the source code. 512-bit operations are known to cause the CPUs
  to run at a lower frequency which can impact performance. This behavior can be
  changed by passing -mprefer-vector-width=512 on the command line.

New Compiler Flags
------------------

- ...

Deprecated Compiler Flags
-------------------------

The following options are deprecated and ignored. They will be removed in
future versions of Clang.

- -mmpx used to enable the __MPX__ preprocessor define for the Intel MPX
  instructions. There were no MPX intrinsics.
- -mno-mpx used to disable -mmpx and is the default behavior.

- ...

Modified Compiler Flags
-----------------------

- ...

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- ...

Windows Support
---------------

- Previous Clang versions contained a work-around to avoid an issue with the
  standard library headers in Visual Studio 2019 versions prior to 16.3. This
  work-around has now been removed, and users of Visual Studio 2019 are
  encouraged to upgrade to 16.3 or later, otherwise they may see link errors as
  below:

  .. code-block:: console

    error LNK2005: "bool const std::_Is_integral<int>" (??$_Is_integral@H@std@@3_NB) already defined




C Language Changes in Clang
---------------------------

- ...

C11 Feature Support
^^^^^^^^^^^^^^^^^^^

...

C++ Language Changes in Clang
-----------------------------

- The behaviour of the `gnu_inline` attribute now matches GCC, for cases
  where used without the `extern` keyword. As this is a change compared to
  how it behaved in previous Clang versions, a warning is emitted for this
  combination.

C++1z Feature Support
^^^^^^^^^^^^^^^^^^^^^

...

Objective-C Language Changes in Clang
-------------------------------------

- ...

OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------

- gcc passes vectors of __int128 in memory on X86-64. Clang historically
  broke the vectors into multiple scalars using two 64-bit values for each
  element. Clang now matches the gcc behavior on Linux and NetBSD. You can
  switch back to old API behavior with flag: -fclang-abi-compat=9.0.

OpenMP Support in Clang
-----------------------

- ...

CUDA Support in Clang
---------------------

- ...

Internal API Changes
--------------------

These are major API changes that have happened since the 9.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

- libTooling APIs that transfer ownership of `FrontendAction` objects now pass
  them by `unique_ptr`, making the ownership transfer obvious in the type
  system. `FrontendActionFactory::create()` now returns a
  `unique_ptr<FrontendAction>`. `runToolOnCode`, `runToolOnCodeWithArgs`,
  `ToolInvocation::ToolInvocation()` now take a `unique_ptr<FrontendAction>`.

Build System Changes
--------------------

These are major changes to the build system that have happened since the 9.0.0
release of Clang. Users of the build system should adjust accordingly.

- In 8.0.0 and below, the install-clang-headers target would install clang's
  resource directory headers. This installation is now performed by the
  install-clang-resource-headers target. Users of the old install-clang-headers
  target should switch to the new install-clang-resource-headers target. The
  install-clang-headers target now installs clang's API headers (corresponding
  to its libraries), which is consistent with the install-llvm-headers target.

- In 9.0.0 and later Clang added a new target, clang-cpp, which generates a
  shared library comprised of all the clang component libraries and exporting
  the clang C++ APIs. Additionally the build system gained the new
  "CLANG_LINK_CLANG_DYLIB" option, which defaults Off, and when set to On, will
  force clang (and clang-based tools) to link the clang-cpp library instead of
  statically linking clang's components. This option will reduce the size of
  binary distributions at the expense of compiler performance.

- ...

AST Matchers
------------

- ...

clang-format
------------

- The ``Standard`` style option specifies which version of C++ should be used
  when parsing and formatting C++ code. The set of allowed values has changed:
  - ``Latest`` will always enable new C++ language features.
  - ``c++03``, ``c++11``, ``c++14``, ``c++17``, ``c++20`` will pin to exactly
    that language version.
  - ``Auto`` is the default and detects style from the code (this is unchanged).
 The previous values of ``Cpp03`` and ``Cpp11`` are deprecated. Note that
 ``Cpp11`` is treated as ``Latest``, as this was always clang-format's behavior.
 (One motivation for this change is the new name describes the behavior better).

libclang
--------

- ...


Static Analyzer
---------------

- ...

.. _release-notes-ubsan:

Undefined Behavior Sanitizer (UBSan)
------------------------------------

- ...

Core Analysis Improvements
==========================

- ...

New Issues Found
================

- ...

Python Binding Changes
----------------------

The following methods have been added:

-  ...

Significant Known Problems
==========================

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Subversion version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us via the `mailing
list <https://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
