=======================================
Clang 8.0.0 (In-Progress) Release Notes
=======================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 8 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 8.0.0. Here we
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

What's New in Clang 8.0.0?
==========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Clang supports use of a profile remapping file, which permits
  profile data captured for one version of a program to be applied
  when building another version where symbols have changed (for
  example, due to renaming a class or namespace).
  See the :doc:`UsersManual` for details.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``-Wextra-semi-stmt`` is a new diagnostic that diagnoses extra semicolons,
  much like ``-Wextra-semi``. This new diagnostic diagnoses all *unnecessary*
  null statements (expression statements without an expression), unless: the
  semicolon directly follows a macro that was expanded to nothing or if the
  semicolon is within the macro itself. This applies to macros defined in system
  headers as well as user-defined macros.

  .. code-block:: c++

      #define MACRO(x) int x;
      #define NULLMACRO(varname)

      void test() {
        ; // <- warning: ';' with no preceding expression is a null statement

        while (true)
          ; // OK, it is needed.

        switch (my_enum) {
        case E1:
          // stuff
          break;
        case E2:
          ; // OK, it is needed.
        }

        MACRO(v0;) // Extra semicolon, but within macro, so ignored.

        MACRO(v1); // <- warning: ';' with no preceding expression is a null statement

        NULLMACRO(v2); // ignored, NULLMACRO expanded to nothing.
      }

- ``-Wempty-init-stmt`` is a new diagnostic that diagnoses empty init-statements
  of ``if``, ``switch``, ``range-based for``, unless: the semicolon directly
  follows a macro that was expanded to nothing or if the semicolon is within the
  macro itself (both macros from system headers, and normal macros). This
  diagnostic is in the ``-Wextra-semi-stmt`` group and is enabled in
  ``-Wextra``.

  .. code-block:: c++

      void test() {
        if(; // <- warning: init-statement of 'if' is a null statement
           true)
          ;

        switch (; // <- warning: init-statement of 'switch' is a null statement
                x) {
          ...
        }

        for (; // <- warning: init-statement of 'range-based for' is a null statement
             int y : S())
          ;
      }


Non-comprehensive list of changes in this release
-------------------------------------------------

- The experimental feature Pretokenized Headers (PTH) was removed in its
  entirely from Clang. The feature did not properly work with about 1/3 of the
  possible tokens available and was unmaintained.

- The internals of libc++ include directory detection on MacOS have changed.
  Instead of running a search based on the ``-resource-dir`` flag, the search
  is now based on the path of the compiler in the filesystem. The default
  behaviour should not change. However, if you override ``-resource-dir``
  manually and rely on the old behaviour you will need to add appropriate
  compiler flags for finding the corresponding libc++ include directory.

New Compiler Flags
------------------

- ``-fprofile-filter-files=[regexes]`` and ``-fprofile-exclude-files=[regexes]``.

  Clang has now options to filter or exclude some files when
  instrumenting for gcov-based profiling.
  See the :doc:`UsersManual` for details.

- ...

Deprecated Compiler Flags
-------------------------

The following options are deprecated and ignored. They will be removed in
future versions of Clang.

- ...

Modified Compiler Flags
-----------------------

- As of clang 8, `alignof` and `_Alignof` return the ABI alignment of a type,
  as opposed to the preferred alignment. `__alignof` still returns the
  preferred alignment. `-fclang-abi-compat=7` (and previous) will make
  `alignof` and `_Alignof` return preferred alignment again.


New Pragmas in Clang
--------------------

- Clang now supports adding multiple `#pragma clang attribute` attributes into
  a scope of pushed attributes.

Attribute Changes in Clang
--------------------------

- ...

Windows Support
---------------

- clang-cl now supports the use of the precompiled header options /Yc and /Yu
  without the filename argument. When these options are used without the
  filename, a `#pragma hdrstop` inside the source marks the end of the
  precompiled code.

- clang-cl has a new command-line option, ``/Zc:dllexportInlines-``, similar to
  ``-fvisibility-inlines-hidden`` on non-Windows, that makes class-level
  `dllexport` and `dllimport` attributes not apply to inline member functions.
  This can significantly reduce compile and link times. See the `User's Manual
  <UsersManual.html#the-zc-dllexportinlines-option>`_ for more info.
- ...


C Language Changes in Clang
---------------------------

- ...

...

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

...

OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------

- `_Alignof` and `alignof` now return the ABI alignment of a type, as opposed
  to the preferred alignment.

  - This is more in keeping with the language of the standards, as well as
    being compatible with gcc
  - `__alignof` and `__alignof__` still return the preferred alignment of
    a type
  - This shouldn't break any ABI except for things that explicitly ask for
    `alignas(alignof(T))`.
  - If you have interfaces that break with this change, you may wish to switch
    to `alignas(__alignof(T))`, instead of using the `-fclang-abi-compat`
    switch.

OpenMP Support in Clang
----------------------------------


CUDA Support in Clang
---------------------


Internal API Changes
--------------------

These are major API changes that have happened since the 7.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

-  ...

AST Matchers
------------

- ...

clang-format
------------


- ...

libclang
--------

...


Static Analyzer
---------------

- ...

...

.. _release-notes-ubsan:

Undefined Behavior Sanitizer (UBSan)
------------------------------------

* The Implicit Conversion Sanitizer (``-fsanitize=implicit-conversion``) group
  was extended. One more type of issues is caught - implicit integer sign change.
  (``-fsanitize=implicit-integer-sign-change``).
  This makes the Implicit Conversion Sanitizer feature-complete,
  with only missing piece being bitfield handling.
  While there is a ``-Wsign-conversion`` diagnostic group that catches this kind
  of issues, it is both noisy, and does not catch **all** the cases.

  .. code-block:: c++

      bool consume(unsigned int val);

      void test(int val) {
        (void)consume(val); // If the value was negative, it is now large positive.
        (void)consume((unsigned int)val); // OK, the conversion is explicit.
      }

  Like some other ``-fsanitize=integer`` checks, these issues are **not**
  undefined behaviour. But they are not *always* intentional, and are somewhat
  hard to track down. This group is **not** enabled by ``-fsanitize=undefined``,
  but the ``-fsanitize=implicit-integer-sign-change`` check
  is enabled by ``-fsanitize=integer``.
  (as is ``-fsanitize=implicit-integer-truncation`` check)

* The Implicit Conversion Sanitizer (``-fsanitize=implicit-conversion``) has
  learned to sanitize compound assignment operators.

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
