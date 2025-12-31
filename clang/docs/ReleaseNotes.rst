.. If you want to modify sections/contents permanently, you should modify both
   ReleaseNotes.rst and ReleaseNotesTemplate.txt.

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

C/C++ Language Potentially Breaking Changes
-------------------------------------------

C++ Specific Potentially Breaking Changes
-----------------------------------------

ABI Changes in This Version
---------------------------

AST Dumping Potentially Breaking Changes
----------------------------------------

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

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++17 Feature Support
^^^^^^^^^^^^^^^^^^^^^

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C Language Changes
------------------

C2y Feature Support
^^^^^^^^^^^^^^^^^^^

C23 Feature Support
^^^^^^^^^^^^^^^^^^^

Non-comprehensive list of changes in this release
-------------------------------------------------

New Compiler Flags
------------------

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
----------------------

Attribute Changes in Clang
--------------------------

Improvements to Clang's diagnostics
-----------------------------------
- Diagnostics messages now refer to ``structured binding`` instead of ``decomposition``,
  to align with `P0615R0 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0615r0.html>`_ changing the term. (#GH157880)
- Clang now suppresses runtime behavior warnings for unreachable code in file-scope
  variable initializers, matching the behavior for functions. This prevents false
  positives for operations in unreachable branches of constant expressions.
- Added a separate diagnostic group ``-Wfunction-effect-redeclarations``, for the more pedantic
  diagnostics for function effects (``[[clang::nonblocking]]`` and ``[[clang::nonallocating]]``).
  Moved the warning for a missing (though implied) attribute on a redeclaration into this group.
  Added a new warning in this group for the case where the attribute is missing/implicit on
  an override of a virtual method.
- Remove ``-Wperf-constraint-implies-noexcept`` from ``-Wall``. This warning is somewhat nit-picky and
  attempts to resolve it, by adding ``noexcept``, can create new ways for programs to crash. (#GH167540)
- Implemented diagnostics when retrieving the tuple size for types where its specialization of `std::tuple_size`
  produces an invalid size (either negative or greater than the implementation limit). (#GH159563)
- Fixed fix-it hint for fold expressions. Clang now correctly places the suggested right
  parenthesis when diagnosing malformed fold expressions. (#GH151787)
- Added fix-it hint for when scoped enumerations require explicit conversions for binary operations. (#GH24265)
- Constant template parameters are now type checked in template definitions,
  including template template parameters.
- Fixed an issue where emitted format-signedness diagnostics were not associated with an appropriate
  diagnostic id. Besides being incorrect from an API standpoint, this was user visible, e.g.:
  "format specifies type 'unsigned int' but the argument has type 'int' [-Wformat]"
  "signedness of format specifier 'u' is incompatible with 'c' [-Wformat]"
  This was misleading, because even though -Wformat is required in order to emit the diagnostics,
  the warning flag the user needs to concerned with here is -Wformat-signedness, which is also
  required and is not enabled by default. With the change you'll now see:
  "format specifies type 'unsigned int' but the argument has type 'int', which differs in signedness [-Wformat-signedness]"
  "signedness of format specifier 'u' is incompatible with 'c' [-Wformat-signedness]"
  and the API-visible diagnostic id will be appropriate.
- Clang now produces better diagnostics for template template parameter matching
  involving 'auto' template parameters.
- Fixed false positives in ``-Waddress-of-packed-member`` diagnostics when
  potential misaligned members get processed before they can get discarded.
  (#GH144729)
- Clang now emits a warning when ``std::atomic_thread_fence`` is used with ``-fsanitize=thread`` as this can
  lead to false positives. (This can be disabled with ``-Wno-tsan``)
- Fix a false positive warning in ``-Wignored-qualifiers`` when the return type is undeduced. (#GH43054)

- Clang now emits a diagnostic with the correct message in case of assigning to const reference captured in lambda. (#GH105647)

- Fixed false positive in ``-Wmissing-noreturn`` diagnostic when it was requiring the usage of
  ``[[noreturn]]`` on lambdas before C++23 (#GH154493).

- Clang now diagnoses the use of ``#`` and ``##`` preprocessor tokens in
  attribute argument lists in C++ when ``-pedantic`` is enabled. The operators
  can be used in macro replacement lists with the usual preprocessor semantics,
  however, non-preprocessor use of tokens now triggers a pedantic warning in C++.
  Compilation in C mode is unchanged, and still permits these tokens to be used. (#GH147217)

- Clang now diagnoses misplaced array bounds on declarators for template
  specializations in th same way as it already did for other declarators.
  (#GH147333)

- A new warning ``-Walloc-size`` has been added to detect calls to functions
  decorated with the ``alloc_size`` attribute don't allocate enough space for
  the target pointer type.

- The :doc:`ThreadSafetyAnalysis` attributes ``ACQUIRED_BEFORE(...)`` and
  ``ACQUIRED_AFTER(...)`` have been moved to the stable feature set and no
  longer require ``-Wthread-safety-beta`` to be used.
- The :doc:`ThreadSafetyAnalysis` gains basic alias-analysis of capability
  pointers under ``-Wthread-safety-beta`` (still experimental), which reduces
  both false positives but also false negatives through more precise analysis.

- Clang now looks through parenthesis for ``-Wundefined-reinterpret-cast`` diagnostic.

- Fixed a bug where the source location was missing when diagnosing ill-formed
  placeholder constraints.

- The two-element, unary mask variant of ``__builtin_shufflevector`` is now
  properly being rejected when used at compile-time. It was not implemented
  and caused assertion failures before (#GH158471).

- Closed a loophole in the diagnosis of function pointer conversions changing
  extended function type information in C mode (#GH41465). Function conversions
  that were previously incorrectly accepted in case of other irrelevant
  conditions are now consistently diagnosed, identical to C++ mode.

- Fix false-positive unused label diagnostic when a label is used in a named break
  or continue (#GH166013)
- Clang now emits a diagnostic in case `vector_size` or `ext_vector_type`
  attributes are used with a negative size (#GH165463).
- Clang no longer emits ``-Wmissing-noreturn`` for virtual methods where
  the function body consists of a `throw` expression (#GH167247).

- A new warning ``-Wenum-compare-typo`` has been added to detect potential erroneous
  comparison operators when mixed with bitwise operators in enum value initializers.
  This can be locally disabled by explicitly casting the initializer value.
- Clang now provides correct caret placement when attributes appear before
  `enum class` (#GH163224).

- A new warning ``-Wshadow-header`` has been added to detect when a header file
  is found in multiple search directories (excluding system paths).

- Clang now detects potential missing format and format_matches attributes on function,
  Objective-C method and block declarations when calling format functions. It is part
  of the format-nonliteral diagnostic (#GH60718)

- Fixed a crash when enabling ``-fdiagnostics-format=sarif`` and the output 
  carries messages like 'In file included from ...' or 'In module ...'.
  Now the include/import locations are written into `sarif.run.result.relatedLocations`.

- Add nested/indented diagnose support when enabling ``-fdiagnostics-format=sarif``.
  Now the notes are recursively mounted under error/warnings, and the diagnostics can
  be parsed into a nested/indented tree.

- Clang now generates a fix-it for C++20 designated initializers when the 
  initializers do not match the declaration order in the structure. 

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a crash when attempting to jump over initialization of a variable with variably modified type. (#GH175540)
- Fixed a crash when using loop hint with a value dependent argument inside a
  generic lambda. (#GH172289)

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

NVPTX Support
^^^^^^^^^^^^^^

X86 Support
^^^^^^^^^^^

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

- DWARF fission is now compatible with linker relaxations, allowing `-gsplit-dwarf` and `-mrelax`
  to be used together when building for the LoongArch platform.

RISC-V Support
^^^^^^^^^^^^^^

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

- Fixed a crash when ``__funcref`` is applied to a non-function pointer type.
  (#GH118233)

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
- Add ``functionTypeLoc`` matcher for matching ``FunctionTypeLoc``.

clang-format
------------

libclang
--------

Code Completion
---------------

Static Analyzer
---------------

New features
^^^^^^^^^^^^

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^

Improvements
^^^^^^^^^^^^

Moved checkers
^^^^^^^^^^^^^^

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------

OpenMP Support
--------------
- Added support for ``transparent`` clause in task and taskloop directives.

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
