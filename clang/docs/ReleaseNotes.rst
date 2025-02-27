===========================================
Clang |release| |ReleaseNotesTitle|
===========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming Clang |version| release.
     Release notes for previous releases can be found on
     `the Releases Page <https://llvm.org/releases/>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release |release|. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. For the libc++ release notes,
see `this page <https://libcxx.llvm.org/ReleaseNotes.html>`_. All LLVM releases
may be downloaded from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Potentially Breaking Changes
============================

- The Objective-C ARC migrator (ARCMigrate) has been removed.

C/C++ Language Potentially Breaking Changes
-------------------------------------------

C++ Specific Potentially Breaking Changes
-----------------------------------------

- The type trait builtin ``__is_referenceable`` has been removed, since it has
  very few users and all the type traits that could benefit from it in the
  standard library already have their own bespoke builtins.

ABI Changes in This Version
---------------------------

- Return larger CXX records in memory instead of using AVX registers. Code compiled with older clang will be incompatible with newer version of the clang unless -fclang-abi-compat=20 is provided. (#GH120670)

AST Dumping Potentially Breaking Changes
----------------------------------------

- Added support for dumping template arguments of structural value kinds.

Clang Frontend Potentially Breaking Changes
-------------------------------------------

Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------

What's New in Clang |release|?
==============================

C++ Language Changes
--------------------

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P1061R10 Structured Bindings can introduce a Pack <https://wg21.link/P1061R10>`_.

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++17 Feature Support
^^^^^^^^^^^^^^^^^^^^^

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The flag `-frelaxed-template-template-args`
  and its negation have been removed, having been deprecated since the previous
  two releases. The improvements to template template parameter matching implemented
  in the previous release, as described in P3310 and P3579, made this flag unnecessary.

- Implemented `CWG2918 Consideration of constraints for address of overloaded `
  `function <https://cplusplus.github.io/CWG/issues/2918.html>`_

C Language Changes
------------------

- Clang now allows an ``inline`` specifier on a typedef declaration of a
  function type in Microsoft compatibility mode. #GH124869
- Clang now allows ``restrict`` qualifier for array types with pointer elements (#GH92847).

C2y Feature Support
^^^^^^^^^^^^^^^^^^^

C23 Feature Support
^^^^^^^^^^^^^^^^^^^

Non-comprehensive list of changes in this release
-------------------------------------------------

New Compiler Flags
------------------

- New option ``-fprofile-continuous`` added to enable continuous profile syncing to file (#GH124353, `docs <https://clang.llvm.org/docs/UsersManual.html#cmdoption-fprofile-continuous>`_).
  The feature has `existed <https://clang.llvm.org/docs/SourceBasedCodeCoverage.html#running-the-instrumented-program>`_)
  for a while and this is just a user facing option.

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
-------------------------

Attribute Changes in Clang
--------------------------
Adding [[clang::unsafe_buffer_usage]] attribute to a method definition now turns off all -Wunsafe-buffer-usage
related warnings within the method body.

- The ``no_sanitize`` attribute now accepts both ``gnu`` and ``clang`` names.
- Clang now diagnoses use of declaration attributes on void parameters. (#GH108819)
- Clang now allows ``__attribute__((model("small")))`` and
  ``__attribute__((model("large")))`` on non-TLS globals in x86-64 compilations.
  This forces the global to be considered small or large in regards to the
  x86-64 code model, regardless of the code model specified for the compilation.

- There is a new ``format_matches`` attribute to complement the existing
  ``format`` attribute. ``format_matches`` allows the compiler to verify that
  a format string argument is equivalent to a reference format string: it is
  useful when a function accepts a format string without its accompanying
  arguments to format. For instance:

  .. code-block:: c

    static int status_code;
    static const char *status_string;

    void print_status(const char *fmt) {
      fprintf(stderr, fmt, status_code, status_string);
      // ^ warning: format string is not a string literal [-Wformat-nonliteral]
    }

    void stuff(void) {
      print_status("%s (%#08x)\n");
      // order of %s and %x is swapped but there is no diagnostic
    }
  
  Before the introducion of ``format_matches``, this code cannot be verified
  at compile-time. ``format_matches`` plugs that hole:

  .. code-block:: c

    __attribute__((format_matches(printf, 1, "%x %s")))
    void print_status(const char *fmt) {
      fprintf(stderr, fmt, status_code, status_string);
      // ^ `fmt` verified as if it was "%x %s" here; no longer triggers
      //   -Wformat-nonliteral, would warn if arguments did not match "%x %s"
    }

    void stuff(void) {
      print_status("%s (%#08x)\n");
      // warning: format specifier 's' is incompatible with 'x'
      // warning: format specifier 'x' is incompatible with 's'
    }

  Like with ``format``, the first argument is the format string flavor and the
  second argument is the index of the format string parameter.
  ``format_matches`` accepts an example valid format string as its third
  argument. For more information, see the Clang attributes documentation.

- Introduced a new statement attribute ``[[clang::atomic]]`` that enables
  fine-grained control over atomic code generation on a per-statement basis.
  Supported options include ``[no_]remote_memory``,
  ``[no_]fine_grained_memory``, and ``[no_]ignore_denormal_mode``. These are
  particularly relevant for AMDGPU targets, where they map to corresponding IR
  metadata.

Improvements to Clang's diagnostics
-----------------------------------

- Improve the diagnostics for deleted default constructor errors for C++ class
  initializer lists that don't explicitly list a class member and thus attempt
  to implicitly default construct that member.
- The ``-Wunique-object-duplication`` warning has been added to warn about objects
  which are supposed to only exist once per program, but may get duplicated when
  built into a shared library.
- Fixed a bug where Clang's Analysis did not correctly model the destructor behavior of ``union`` members (#GH119415).
- A statement attribute applied to a ``case`` label no longer suppresses
  'bypassing variable initialization' diagnostics (#84072).
- The ``-Wunsafe-buffer-usage`` warning has been updated to warn
  about unsafe libc function calls.  Those new warnings are emitted
  under the subgroup ``-Wunsafe-buffer-usage-in-libc-call``.
- Diagnostics on chained comparisons (``a < b < c``) are now an error by default. This can be disabled with
  ``-Wno-error=parentheses``.

- The :doc:`ThreadSafetyAnalysis` now supports ``-Wthread-safety-pointer``,
  which enables warning on passing or returning pointers to guarded variables
  as function arguments or return value respectively. Note that
  :doc:`ThreadSafetyAnalysis` still does not perform alias analysis. The
  feature will be default-enabled with ``-Wthread-safety`` in a future release.

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------

- Clang now outputs correct values when #embed data contains bytes with negative
  signed char values (#GH102798).

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The behvaiour of ``__add_pointer`` and ``__remove_pointer`` for Objective-C++'s ``id`` and interfaces has been fixed.

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 - Fixed crash when a parameter to the ``clang::annotate`` attribute evaluates to ``void``. See #GH119125

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Clang is now better at keeping track of friend function template instance contexts. (#GH55509)
- Clang now prints the correct instantiation context for diagnostics suppressed
  by template argument deduction.
- The initialization kind of elements of structured bindings
  direct-list-initialized from an array is corrected to direct-initialization.
- Clang no longer crashes when a coroutine is declared ``[[noreturn]]``. (#GH127327)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed type checking when a statement expression ends in an l-value of atomic type. (#GH106576)

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

- HTML tags in comments that span multiple lines are now parsed correctly by Clang's comment parser. (#GH120843)

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

NVPTX Support
^^^^^^^^^^^^^^

Hexagon Support
^^^^^^^^^^^^^^^

-  The default compilation target has been changed from V60 to V68.

X86 Support
^^^^^^^^^^^

- Disable ``-m[no-]avx10.1`` and switch ``-m[no-]avx10.2`` to alias of 512 bit
  options.
- Change ``-mno-avx10.1-512`` to alias of ``-mno-avx10.1-256`` to disable both
  256 and 512 bit instructions.

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^

- Add support for `-mtune=generic-ooo` (a generic out-of-order model).

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA Support
^^^^^^^^^^^^

AIX Support
^^^^^^^^^^^

NetBSD Support
^^^^^^^^^^^^^^

WebAssembly Support
^^^^^^^^^^^^^^^^^^^

AVR Support
^^^^^^^^^^^

DWARF Support in Clang
----------------------

Floating Point Support in Clang
-------------------------------

Fixed Point Support in Clang
----------------------------

AST Matchers
------------

clang-format
------------

- Adds ``BreakBeforeTemplateCloser`` option.
- Adds ``BinPackLongBracedList`` option to override bin packing options in
  long (20 item or more) braced list initializer lists.
- Add the C language instead of treating it like C++.
- Allow specifying the language (C, C++, or Objective-C) for a ``.h`` file by
  adding a special comment (e.g. ``// clang-format Language: ObjC``) near the
  top of the file.

libclang
--------

- Fixed a buffer overflow in ``CXString`` implementation. The fix may result in
  increased memory allocation.

Code Completion
---------------

Static Analyzer
---------------

New features
^^^^^^^^^^^^

A new flag - `-static-libclosure` was introduced to support statically linking
the runtime for the Blocks extension on Windows. This flag currently only
changes the code generation, and even then, only on Windows. This does not
impact the linker behaviour like the other `-static-*` flags.

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^

Improvements
^^^^^^^^^^^^

Moved checkers
^^^^^^^^^^^^^^

- After lots of improvements, the checker ``alpha.security.ArrayBoundV2`` is
  renamed to ``security.ArrayBound``. As this checker is stable now, the old
  checker ``alpha.security.ArrayBound`` (which was searching for the same kind
  of bugs with an different, simpler and less accurate algorithm) is removed.

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------

OpenMP Support
--------------
- Added support 'no_openmp_constructs' assumption clause.
- Added support for 'omp stripe' directive.

Improvements
^^^^^^^^^^^^

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us on the `Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_.
