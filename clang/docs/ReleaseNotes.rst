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

- The ``__has_builtin`` function now only considers the currently active target when being used with target offloading.

C++ Specific Potentially Breaking Changes
-----------------------------------------
- For C++20 modules, the Reduced BMI mode will be the default option. This may introduce
  regressions if your build system supports two-phase compilation model but haven't support
  reduced BMI or it is a compiler bug or a bug in users code.

- Clang now correctly diagnoses during constant expression evaluation undefined behavior due to member
  pointer access to a member which is not a direct or indirect member of the most-derived object
  of the accessed object but is instead located directly in a sibling class to one of the classes
  along the inheritance hierarchy of the most-derived object as ill-formed.
  Other scenarios in which the member is not member of the most derived object were already
  diagnosed previously. (#GH150709)

  .. code-block:: c++

    struct A {};
    struct B : A {};
    struct C : A { constexpr int foo() const { return 1; } };
    constexpr A a;
    constexpr B b;
    constexpr C c;
    constexpr auto mp = static_cast<int(A::*)() const>(&C::foo);
    static_assert((a.*mp)() == 1); // continues to be rejected
    static_assert((b.*mp)() == 1); // newly rejected
    static_assert((c.*mp)() == 1); // accepted

ABI Changes in This Version
---------------------------

AST Dumping Potentially Breaking Changes
----------------------------------------
- How nested name specifiers are dumped and printed changes, keeping track of clang AST changes.

Clang Frontend Potentially Breaking Changes
-------------------------------------------
- Members of anonymous unions/structs are now injected as ``IndirectFieldDecl``
  into the enclosing record even if their names conflict with other names in the
  scope. These ``IndirectFieldDecl`` are marked invalid.

Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------
- TypeKind ``ELABORATED`` is not used anymore, per clang AST changes removing
  ElaboratedTypes. The value becomes unused, and all the existing users should
  expect the former underlying type to be reported instead.

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
- Added ``__builtin_elementwise_fshl`` and ``__builtin_elementwise_fshr``.

- ``__builtin_elementwise_abs`` can now be used in constant expression.

- Added ``__builtin_elementwise_minnumnum`` and ``__builtin_elementwise_maxnumnum``.

- Trapping UBSan (e.g. ``-fsanitize-trap=undefined``) now emits a string describing the reason for
  trapping into the generated debug info. This feature allows debuggers (e.g. LLDB) to display
  the reason for trapping if the trap is reached. The string is currently encoded in the debug
  info as an artificial frame that claims to be inlined at the trap location. The function used
  for the artificial frame is an artificial function whose name encodes the reason for trapping.
  The encoding used is currently the same as ``__builtin_verbose_trap`` but might change in the future.
  This feature is enabled by default but can be disabled by compiling with
  ``-fno-sanitize-annotate-debug-info-traps``.

- ``__builtin_elementwise_max`` and ``__builtin_elementwise_min`` functions for integer types can
  now be used in constant expressions.

- Use of ``__has_feature`` to detect the ``ptrauth_qualifier`` and ``ptrauth_intrinsics``
  features has been deprecated, and is restricted to the arm64e target only. The
  correct method to check for these features is to test for the ``__PTRAUTH__``
  macro.


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
- Fixed fix-it hint for fold expressions. Clang now correctly places the suggested right
  parenthesis when diagnosing malformed fold expressions. (#GH151787)
- ``-Wstring-concatenation`` now diagnoses every missing comma in an initializer list,
  rather than stopping after the first. (#GH153745)

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
  
- Fixed false positives in ``-Waddress-of-packed-member`` diagnostics when
  potential misaligned members get processed before they can get discarded.
  (#GH144729)

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
- Fixed a crash with incompatible pointer to integer conversions in designated
  initializers involving string literals. (#GH154046)

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fix an ambiguous reference to the builtin `type_info` (available when using
  `-fms-compatibility`) with modules. (#GH38400)

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``[[nodiscard]]`` is now respected on Objective-C and Objective-C++ methods.
  (#GH141504)
- Fixes some late parsed attributes, when applied to function definitions, not being parsed
  in function try blocks, and some situations where parsing of the function body
  is skipped, such as error recovery and code completion. (#GH153551)
- Using ``[[gnu::cleanup(some_func)]]`` where some_func is annotated with
  ``[[gnu::error("some error")]]`` now correctly triggers an error. (#GH146520)

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^
- Diagnose binding a reference to ``*nullptr`` during constant evaluation. (#GH48665)
- Suppress ``-Wdeprecated-declarations`` in implicitly generated functions. (#GH147293)
- Fix a crash when deleting a pointer to an incomplete array (#GH150359).
- Fixed a mismatched lambda scope bug when propagating up ``consteval`` within nested lambdas. (#GH145776)
- Fix an assertion failure when expression in assumption attribute
  (``[[assume(expr)]]``) creates temporary objects.
- Fix the dynamic_cast to final class optimization to correctly handle
  casts that are guaranteed to fail (#GH137518).
- Fix bug rejecting partial specialization of variable templates with auto NTTPs (#GH118190).
- Fix a crash if errors "member of anonymous [...] redeclares" and
  "intializing multiple members of union" coincide (#GH149985).
- Fix a crash when using ``explicit(bool)`` in pre-C++11 language modes. (#GH152729)
- Fix the parsing of variadic member functions when the ellipis immediately follows a default argument.(#GH153445)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fix incorrect name qualifiers applied to alias CTAD. (#GH136624)
- Fixed ElaboratedTypes appearing within NestedNameSpecifier, which was not a
  legal representation. This is fixed because ElaboratedTypes don't exist anymore. (#GH43179) (#GH68670) (#GH92757)
- Fix unrecognized html tag causing undesirable comment lexing (#GH152944)
- Fix comment lexing of special command names (#GH152943)

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

- Add support for `__attribute__((interrupt("rnmi")))` to be used with the `Smrnmi` extension.
  With this the `Smrnmi` extension is fully supported.

- Add `-march=unset` to clear any previous `-march=` value. This ISA string will
  be computed from `-mcpu` or the platform default.

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
- Removed elaboratedType matchers, and related nested name specifier changes,
  following the corresponding changes in the clang AST.
- Ensure ``hasBitWidth`` doesn't crash on bit widths that are dependent on template
  parameters.

- Add a boolean member ``IgnoreSystemHeaders`` to ``MatchFinderOptions``. This
  allows it to ignore nodes in system headers when traversing the AST.

clang-format
------------
- Add ``SpaceInEmptyBraces`` option and set it to ``Always`` for WebKit style.

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
- Fixed a crash when parsing ``#embed`` parameters with unmatched closing brackets. (#GH152829)

Improvements
^^^^^^^^^^^^

Moved checkers
^^^^^^^^^^^^^^

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------
- Exposed `clang_getCursorLanguage` via `Cursor.language`.

OpenMP Support
--------------
- Added parsing and semantic analysis support for the ``need_device_addr``
  modifier in the ``adjust_args`` clause.
- Allow array length to be omitted in array section subscript expression.
- Fixed non-contiguous strided update in the ``omp target update`` directive with the ``from`` clause.

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
