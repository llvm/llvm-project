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

- The ``__has_builtin`` function now only considers the currently active target when being used with target offloading.

C++ Specific Potentially Breaking Changes
-----------------------------------------
- For C++20 modules, the Reduced BMI mode will be the default option. This may introduce
  regressions if your build system supports two-phase compilation model but haven't support
  reduced BMI or it is a compiler bug or a bug in users code.

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
- Added ``__builtin_elementwise_minnumnum`` and ``__builtin_elementwise_maxnumnum``.

- Trapping UBSan (e.g. ``-fsanitize-trap=undefined``) now emits a string describing the reason for 
  trapping into the generated debug info. This feature allows debuggers (e.g. LLDB) to display 
  the reason for trapping if the trap is reached. The string is currently encoded in the debug 
  info as an artificial frame that claims to be inlined at the trap location. The function used 
  for the artificial frame is an artificial function whose name encodes the reason for trapping. 
  The encoding used is currently the same as ``__builtin_verbose_trap`` but might change in the future. 
  This feature is enabled by default but can be disabled by compiling with 
  ``-fno-sanitize-annotate-debug-info-traps``.

New Compiler Flags
------------------
- New option ``-fno-sanitize-annotate-debug-info-traps`` added to disable emitting trap reasons into the debug info when compiling with trapping UBSan (e.g. ``-fsanitize-trap=undefined``).

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
-------------------------

Attribute Changes in Clang
--------------------------

Improvements to Clang's diagnostics
-----------------------------------
- Added a separate diagnostic group ``-Wfunction-effect-redeclarations``, for the more pedantic
  diagnostics for function effects (``[[clang::nonblocking]]`` and ``[[clang::nonallocating]]``).
  Moved the warning for a missing (though implied) attribute on a redeclaration into this group.
  Added a new warning in this group for the case where the attribute is missing/implicit on
  an override of a virtual method.

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------
- Fix a crash when marco name is empty in ``#pragma push_macro("")`` or
  ``#pragma pop_macro("")``. (#GH149762).
- `-Wunreachable-code`` now diagnoses tautological or contradictory
  comparisons such as ``x != 0 || x != 1.0`` and ``x == 0 && x == 1.0`` on
  targets that treat ``_Float16``/``__fp16`` as native scalar types. Previously
  the warning was silently lost because the operands differed only by an implicit
  cast chain. (#GH149967).

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``[[nodiscard]]`` is now respected on Objective-C and Objective-C++ methods.
  (#GH141504)

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^
- Diagnose binding a reference to ``*nullptr`` during constant evaluation. (#GH48665)
- Suppress ``-Wdeprecated-declarations`` in implicitly generated functions. (#GH147293)
- Fix a crash when deleting a pointer to an incomplete array (#GH150359).
- Fix an assertion failure when expression in assumption attribute
  (``[[assume(expr)]]``) creates temporary objects.

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

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
- Ensure ``hasBitWidth`` doesn't crash on bit widths that are dependent on template
  parameters.

clang-format
------------

libclang
--------

Code Completion
---------------

Static Analyzer
---------------
- The Clang Static Analyzer now handles parenthesized initialization.
  (#GH148875)
- ``__datasizeof`` (C++) and ``_Countof`` (C) no longer cause a failed assertion
  when given an operand of VLA type. (#GH151711)

New features
^^^^^^^^^^^^

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^
- Fixed a crash in the static analyzer that when the expression in an 
  ``[[assume(expr)]]`` attribute was enclosed in parentheses.  (#GH151529)

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
- Added parsing and semantic analysis support for the ``need_device_addr``
  modifier in the ``adjust_args`` clause.
- Allow array length to be omitted in array section subscript expression.

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
