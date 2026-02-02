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
- Remove ``CompletionString.Availability``. No libclang interfaces returned instances of it.
- ``CompletionString.availability`` now returns instances of ``CompletionString.AvailabilityKindCompat``.

  Instances of ``AvailabilityKindCompat`` have the same ``__str__`` representation
  as the previous ``CompletionChunk.Kind`` and are equality-comparable with
  the existing ``AvailabilityKind`` enum. It will be replaced by ``AvailabilityKind``
  in a future release. When this happens, the return type of ``CompletionString.availability``
  will change to ``AvailabilityKind``, so it is recommended to use ``AvailabilityKind``
  to compare with the return values of ``CompletionString.availability``.
- Remove ``availabilityKinds``. In this release, uses of ``availabilityKinds``
  need to be replaced by ``CompletionString.AvailabilityKind``.
- ``CompletionChunk.kind`` now returns instances of ``CompletionChunkKind``.

  Instances of ``CompletionChunkKind`` have the same ``__str__`` representation
  as the previous ``CompletionChunk.Kind`` for compatibility.
  These representations will be changed in a future release to match other enums.
- Remove ``completionChunkKindMap``. In this release, uses of ``completionChunkKindMap``
  need to be replaced by ``CompletionChunkKind``.
- Move ``SPELLING_CACHE`` into ``CompletionChunk`` and change it to use
  ``CompletionChunkKind`` instances as keys, instead of the enum values.
  An alias is kept in the form of a ``SPELLING_CACHE`` variable, but it only supports
  ``__getitem__`` and ``__contains__``. It will be removed in a future release.
  Please migrate to using ``CompletionChunk.SPELLING_CACHE`` instead.

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

- Clang now supports `P1857R3 <https://wg21.link/p1857r3>`_ Modules Dependency Discovery. (#GH54047)

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

- Added ``__builtin_stdc_rotate_left`` and ``__builtin_stdc_rotate_right``
  for bit rotation of unsigned integers including ``_BitInt`` types. Rotation
  counts are normalized modulo the bit-width and support negative values.
  Usable in constant expressions. Implicit conversion is supported for
  class/struct types with conversion operators.

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
- Added ``-Wlifetime-safety`` to enable lifetime safety analysis,
  a CFG-based intra-procedural analysis that detects use-after-free and related
  temporal safety bugs. See the
  `RFC <https://discourse.llvm.org/t/rfc-intra-procedural-lifetime-analysis-in-clang/86291>`_
  for more details. By design, this warning is enabled in ``-Wall``. To disable
  the analysis, use ``-Wno-lifetime-safety`` or ``-fno-lifetime-safety``.

- Added ``-Wlifetime-safety-suggestions`` to enable lifetime annotation suggestions.
  This provides suggestions for function parameters that
  should be marked ``[[clang::lifetimebound]]`` based on lifetime analysis. For
  example, for the following function:

  .. code-block:: c++

    int* p(int *in) { return in; }

  Clang will suggest:

  .. code-block:: c++

    warning: parameter in intra-TU function should be marked [[clang::lifetimebound]]
    int* p(int *in) { return in; }
           ^~~~~~~
                   [[clang::lifetimebound]]
    note: param returned here
    int* p(int *in) { return in; }
                             ^~

- Added ``-Wlifetime-safety-noescape`` to detect misuse of ``[[clang::noescape]]``
  annotation where the parameter escapes through return. For example:

  .. code-block:: c++

    int* p(int *in [[clang::noescape]]) { return in; }

  Clang will warn:

  .. code-block:: c++

    warning: parameter is marked [[clang::noescape]] but escapes
    int* p(int *in [[clang::noescape]]) { return in; }
           ^~~~~~~
    note: returned here
    int* p(int *in [[clang::noescape]]) { return in; }
                                                 ^~

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------
- Fixed a failed assertion in the preprocessor when ``__has_embed`` parameters are missing parentheses. (#GH175088)

- Fix lifetime extension of temporaries in for-range-initializers in templates. (#GH165182)

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed a behavioral discrepancy between deleted functions and private members when checking the ``enable_if`` attribute. (#GH175895)

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed a crash when instantiating ``requires`` expressions involving substitution failures in C++ concepts. (#GH176402)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a crash when attempting to jump over initialization of a variable with variably modified type. (#GH175540)
- Fixed a crash when using loop hint with a value dependent argument inside a
  generic lambda. (#GH172289)
- Fixed a crash in C++ overload resolution with ``_Atomic``-qualified argument types. (#GH170433)

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

- Initial support for gfx1310

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

SystemZ Support
^^^^^^^^^^^^^^^

- Add support for `#pragma export` for z/OS.  This is a pragma used to export functions and variables
  with external linkage from shared libraries.  It provides compatibility with the IBM XL C/C++
  compiler.

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

.. comment:
  This is for the Static Analyzer.
  Using the caret `^^^` underlining for subsections:
    - Crash and bug fixes
    - New checkers and features
    - Improvements
    - Moved checkers

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------
- Add deprecation warnings to ``CompletionChunk.isKind...`` methods.
  These will be removed in a future release. Existing uses should be adapted
  to directly compare equality of the ``CompletionChunk`` kind with
  the corresponding ``CompletionChunkKind`` variant.

  Affected methods: ``isKindOptional``, ``isKindTypedText``, ``isKindPlaceHolder``,
  ``isKindInformative`` and ``isKindResultType``.

OpenMP Support
--------------
- Added support for ``transparent`` clause in task and taskloop directives.
- Added support for ``use_device_ptr`` clause to accept an optional
  ``fallback`` modifier (``fb_nullify`` or ``fb_preserve``) with OpenMP >= 61.

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
