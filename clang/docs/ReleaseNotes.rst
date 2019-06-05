=======================================
Clang 9.0.0 (In-Progress) Release Notes
=======================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 9 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 9.0.0. Here we
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

What's New in Clang 9.0.0?
==========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- ...

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ...

Non-comprehensive list of changes in this release
-------------------------------------------------

- ...


New Compiler Flags
------------------

- ...

Deprecated Compiler Flags
-------------------------

The following options are deprecated and ignored. They will be removed in
future versions of Clang.

- ...

Modified Compiler Flags
-----------------------

- ``clang -dumpversion`` now returns the version of Clang itself.

- ...

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- ...

Windows Support
---------------

- clang-cl now treats non-existent files as possible typos for flags,
  ``clang-cl /diagnostic:caret /c test.cc`` for example now produces
  ``clang: error: no such file or directory: '/diagnostic:caret'; did you mean '/diagnostics:caret'?``



C Language Changes in Clang
---------------------------

- ``__FILE_NAME__`` macro has been added as a Clang specific extension supported
  in all C-family languages. This macro is similar to ``__FILE__`` except it
  will always provide the last path component when possible.

- ...

C11 Feature Support
^^^^^^^^^^^^^^^^^^^

...

C++ Language Changes in Clang
-----------------------------

- ...

C++1z Feature Support
^^^^^^^^^^^^^^^^^^^^^

...

Objective-C Language Changes in Clang
-------------------------------------

- Fixed encoding of ObjC pointer types that are pointers to typedefs.

.. code-block:: c++

      typedef NSArray<NSObject *> MyArray;

      // clang used to encode this as "^{NSArray=#}" instead of "@".
      const char *s0 = @encode(MyArray *);

OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------

- ...

OpenMP Support in Clang
-----------------------

- Added emission of the debug information for NVPTX target devices.

CUDA Support in Clang
---------------------

- Added emission of the debug information for the device code.

Internal API Changes
--------------------

These are major API changes that have happened since the 8.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

Build System Changes
--------------------

These are major changes to the build system that have happened since the 8.0.0
release of Clang. Users of the build system should adjust accordingly.

- In 8.0.0 and below, the install-clang-headers target would install clang's
  resource directory headers. This installation is now performed by the
  install-clang-resource-headers target. Users of the old install-clang-headers
  target should switch to the new install-clang-resource-headers target. The
  install-clang-headers target now installs clang's API headers (corresponding
  to its libraries), which is consistent with the install-llvm-headers target.

-  ...

AST Matchers
------------

- ...

clang-format
------------

- Add language support for clang-formatting C# files
- Add Microsoft coding style to encapsulate default C# formatting style
- Added new option `PPDIS_BeforeHash` (in configuration: `BeforeHash`) to
  `IndentPPDirectives` which indents preprocessor directives before the hash.

libclang
--------

- When `CINDEXTEST_INCLUDE_ATTRIBUTED_TYPES` is not provided when making a
  CXType, the equivalent type of the AttributedType is returned instead of the
  modified type if the user does not want attribute sugar. The equivalent type
  represents the minimally-desugared type which the AttributedType is
  canonically equivalent to.


Static Analyzer
---------------

- The UninitializedObject checker is now considered as stable.
  (moved from the 'alpha.cplusplus' to the 'optin.cplusplus' package)

...

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
