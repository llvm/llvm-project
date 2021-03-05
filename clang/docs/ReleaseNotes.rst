========================================
Clang 12.0.0 (In-Progress) Release Notes
========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 12 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 12.0.0. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Clang 12.0.0?
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

- ...

Non-comprehensive list of changes in this release
-------------------------------------------------

- The builtin intrinsics ``__builtin_bitreverse8``, ``__builtin_bitreverse16``,
  ``__builtin_bitreverse32`` and ``__builtin_bitreverse64`` may now be used
  within constant expressions.

- The builtin intrinsics ``__builtin_rotateleft8``, ``__builtin_rotateleft16``,
  ``__builtin_rotateleft32`` and ``__builtin_rotateleft64`` may now be used
  within constant expressions.

- The builtin intrinsics ``__builtin_rotateright8``, ``__builtin_rotateright16``,
  ``__builtin_rotateright32`` and ``__builtin_rotateright64`` may now be used
  within constant expressions.

New Compiler Flags
------------------

- ...

- AArch64 options ``-moutline-atomics``, ``-mno-outline-atomics`` to enable
  and disable calls to helper functions implementing atomic operations. These
  out-of-line helpers like '__aarch64_cas8_relax' will detect at runtime
  AArch64 Large System Extensions (LSE) availability and either use their
  atomic instructions, or falls back to LL/SC loop. These options do not apply
  if the compilation target supports LSE. Atomic instructions are used directly
  in that case. The option's behaviour mirrors GCC, the helpers are implemented
  both in compiler-rt and libgcc.

- New option ``-fbinutils-version=`` specifies the targeted binutils version.
  For example, ``-fbinutils-version=2.35`` means compatibility with GNU as/ld
  before 2.35 is not needed: new features can be used and there is no need to
  work around old GNU as/ld bugs.

Deprecated Compiler Flags
-------------------------

The following options are deprecated and ignored. They will be removed in
future versions of Clang.

- The clang-cl ``/fallback`` flag, which made clang-cl invoke Microsoft Visual
  C++ on files it couldn't compile itself, has been deprecated. It will be
  removed in Clang 13.

- ...

Modified Compiler Flags
-----------------------

- On ELF, ``-gz`` now defaults to ``-gz=zlib`` with the integrated assembler.
  It produces ``SHF_COMPRESSED`` style compression of debug information. GNU
  binutils 2.26 or newer, or lld is required to link produced object files. Use
  ``-gz=zlib-gnu`` to get the old behavior.
- Now that `this` pointers are tagged with `nonnull` and `dereferenceable(N)`,
  `-fno-delete-null-pointer-checks` has gained the power to remove the
  `nonnull` attribute on `this` for configurations that need it to be nullable.
- ``-gsplit-dwarf`` no longer implies ``-g2``.
- ``-fasynchronous-unwind-tables`` is now the default on Linux AArch64/PowerPC.
  This behavior matches newer GCC.
  (`D91760 <https://reviews.llvm.org/D91760>`_)
  (`D92054 <https://reviews.llvm.org/D92054>`_)
- Support has been added for the following processors (command-line identifiers
  in parentheses):

  - Arm Cortex-A78C (cortex-a78c).
  - Arm Cortex-R82 (cortex-r82).
  - Arm Neoverse V1 (neoverse-v1).
  - Arm Neoverse N2 (neoverse-n2).
  - Fujitsu A64FX (a64fx).
  For example, to select architecture support and tuning for Neoverse-V1 based
  systems, use ``-mcpu=neoverse-v1``.

Removed Compiler Flags
-------------------------

The following options no longer exist.

- clang-cl's ``/Zd`` flag no longer exist. But ``-gline-tables-only`` still
  exists and does the same thing.

New Pragmas in Clang
--------------------

- ...

Modified Pragmas in Clang
-------------------------

- The "#pragma clang loop vectorize_width" has been extended to support an
  optional 'fixed|scalable' argument, which can be used to indicate that the
  compiler should use fixed-width or scalable vectorization.  Fixed-width is
  assumed by default.

  Scalable or vector length agnostic vectorization is an experimental feature
  for targets that support scalable vectors. For more information please refer
  to the Clang Language Extensions documentation.

Attribute Changes in Clang
--------------------------

- Added support for the C++20 likelihood attributes ``[[likely]]`` and
  ``[[unlikely]]``. As an extension they can be used in C++11 and newer.
  This extension is enabled by default.

Windows Support
---------------

- Implicitly add ``.exe`` suffix for MinGW targets, even when cross compiling.
  (This matches a change from GCC 8.)

- Windows on Arm64: programs using the C standard library's setjmp and longjmp
  functions may crash with a "Security check failure or stack buffer overrun"
  exception. To workaround (with reduced security), compile with
  /guard:cf,nolongjmp.

- Windows on Arm64: LLVM 12 adds official binary release hosted on
  Windows on Arm64.  The binary is built and tested by Linaro alongside
  AArch64 and ARM 32-bit Linux binary releases.  This first WoA release
  includes Clang compiler, LLD Linker, and compiler-rt runtime libraries.
  Work on LLDB, sanitizer support, OpenMP, and other features is in progress
  and will be included in future Windows on Arm64 LLVM releases.

C Language Changes in Clang
---------------------------

- ...

C++ Language Changes in Clang
-----------------------------

- ...

C++1z Feature Support
^^^^^^^^^^^^^^^^^^^^^
...

Objective-C Language Changes in Clang
-------------------------------------

OpenCL Kernel Language Changes in Clang
---------------------------------------

- Improved online documentation: :doc:`UsersManual` and :doc:`OpenCLSupport`
  pages.
- Added ``-cl-std=CL3.0`` and predefined version macro for OpenCL 3.0.
- Added ``-cl-std=CL1.0`` and mapped to the existing OpenCL 1.0 functionality.
- Improved OpenCL extension handling per target.
- Added clang extension for function pointers ``__cl_clang_function_pointers``
  and variadic functions ``__cl_clang_variadic_functions``, more details can be
  found in :doc:`LanguageExtensions`.
- Removed extensions without kernel language changes:
  ``cl_khr_select_fprounding_mode``, ``cl_khr_gl_sharing``, ``cl_khr_icd``,
  ``cl_khr_gl_event``, ``cl_khr_d3d10_sharing``, ``cl_khr_context_abort``,
  ``cl_khr_d3d11_sharing``, ``cl_khr_dx9_media_sharing``,
  ``cl_khr_image2d_from_buffer``, ``cl_khr_initialize_memory``,
  ``cl_khr_gl_depth_images``, ``cl_khr_spir``, ``cl_khr_egl_event``,
  ``cl_khr_egl_image``, ``cl_khr_terminate_context``.
- Improved diagnostics for  unevaluated ``vec_step`` expression.
- Allow nested pointers (e.g. pointer-to-pointer) kernel arguments beyond OpenCL
  1.2.
- Added ``global_device`` and ``global_host`` address spaces for USM
  allocations.

Miscellaneous improvements in C++ for OpenCL support:

- Added diagnostics for pointers to member functions and references to
  functions.
- Added support of ``vec_step`` builtin.
- Fixed ICE on address spaces with forwarding references and templated copy
  constructors.
- Removed warning for variadic macro use.

ABI Changes in Clang
--------------------

OpenMP Support in Clang
-----------------------

- ...

CUDA Support in Clang
---------------------

- ...

X86 Support in Clang
--------------------

- The x86 intrinsics ``_mm_popcnt_u32``, ``_mm_popcnt_u64``, ``_popcnt32``,
  ``_popcnt64``, ``__popcntd`` and ``__popcntq``  may now be used within
  constant expressions.

- The x86 intrinsics ``_bit_scan_forward``, ``__bsfd`` and ``__bsfq`` may now
  be used within constant expressions.

- The x86 intrinsics ``_bit_scan_reverse``, ``__bsrd`` and ``__bsrq`` may now
  be used within constant expressions.

- The x86 intrinsics ``__bswap``, ``__bswapd``, ``__bswap64`` and ``__bswapq``
  may now be used within constant expressions.

- The x86 intrinsics ``_castf32_u32``, ``_castf64_u64``, ``_castu32_f32`` and
  ``_castu64_f64`` may now be used within constant expressions.

- The x86 intrinsics ``__rolb``, ``__rolw``, ``__rold``, ``__rolq`, ``_rotl``,
  ``_rotwl`` and ``_lrotl`` may now be used within constant expressions.

- The x86 intrinsics ``__rorb``, ``__rorw``, ``__rord``, ``__rorq`, ``_rotr``,
  ``_rotwr`` and ``_lrotr`` may now be used within constant expressions.

- Support for ``-march=alderlake``, ``-march=sapphirerapids`` and
  ``-march=znver3`` was added.

- Support for ``-march=x86-64-v[234]`` has been added.
  See :doc:`UsersManual` for details about these micro-architecture levels.

- The -mtune command line option is no longer ignored for X86. This can be used
  to request microarchitectural optimizations independent on -march. -march=<cpu>
  implies -mtune=<cpu>. -mtune=generic is the default with no -march or -mtune
  specified.

- Support for ``HRESET`` instructions has been added.

- Support for ``UINTR`` instructions has been added.

- Support for ``AVXVNNI`` instructions has been added.

Internal API Changes
--------------------

These are major API changes that have happened since the 11.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

- ...

Build System Changes
--------------------

These are major changes to the build system that have happened since the 11.0.0
release of Clang. Users of the build system should adjust accordingly.

- ...

AST Matchers
------------

- The ``mapAnyOf()`` matcher was added. This allows convenient matching of
  different AST nodes which have a compatible matcher API. For example,
  ``mapAnyOf(ifStmt, forStmt).with(hasCondition(integerLiteral()))``
  matches any ``IfStmt`` or ``ForStmt`` with a integer literal as the
  condition.

- The ``binaryOperation()`` matcher allows matching expressions which
  appear like binary operators in the code, even if they are really
  ``CXXOperatorCallExpr`` for example. It is based on the ``mapAnyOf()``
  matcher functionality. The matcher API for the latter node has been
  extended with ``hasLHS()`` etc to facilitate the abstraction.

- Matcher API for ``CXXRewrittenBinaryOperator`` has been added. In addition
  to explicit matching with the ``cxxRewrittenBinaryOperator()`` matcher, the
  ``binaryOperation()`` matches on nodes of this type.

- The behavior of ``TK_IgnoreUnlessSpelledInSource`` with the ``traverse()``
  matcher has been changed to no longer match on template instantiations or on
  implicit nodes which are not spelled in the source.

- The ``TK_IgnoreImplicitCastsAndParentheses`` traversal kind was removed. It
  is recommended to use ``TK_IgnoreUnlessSpelledInSource`` instead.

- The behavior of the ``forEach()`` matcher was changed to not internally
  ignore implicit and parenthesis nodes.  This makes it consistent with
  the ``has()`` matcher.  Uses of ``forEach()`` relying on the old behavior
  can now use the  ``traverse()`` matcher or ``ignoringParenCasts()``.

- Several AST Matchers have been changed to match based on the active
  traversal mode.  For example, ``argumentCountIs()`` matches the number of
  arguments written in the source, ignoring default arguments represented
  by ``CXXDefaultArgExpr`` nodes.

- Improvements in AST Matchers allow more matching of template declarations,
  independent of their template instantations.

clang-format
------------

- Option ``BitFieldColonSpacing`` has been added that decides how
  space should be added around identifier, colon and bit-width in
  bitfield definitions.

  .. code-block:: c++

    // Both (default)
    struct F {
      unsigned dscp : 6;
      unsigned ecn  : 2; // AlignConsecutiveBitFields=true
    };
    // None
    struct F {
      unsigned dscp:6;
      unsigned ecn :2;
    };
    // Before
    struct F {
      unsigned dscp :6;
      unsigned ecn  :2;
    };
    // After
    struct F {
      unsigned dscp: 6;
      unsigned ecn : 2;
    };


- Experimental Support in clang-format for concepts has been improved, to
  aid this the follow options have been added

- Option ``IndentRequires`` has been added to indent the ``requires`` keyword
  in templates.

- Option ``BreakBeforeConceptDeclarations`` has been added to aid the formatting of concepts.

- Option ``IndentPragmas`` has been added to allow #pragma to indented with the current scope
  level. This is especially useful when using #pragma to mark OpenMP sections of code.

- Option ``SpaceBeforeCaseColon`` has been added to add a space before the
  colon in a case or default statement.

- Option ``StatementAttributeLikeMacros`` has been added to declare
  macros which are not parsed as a type in front of a statement. See
  the documentation for an example.

- Options ``AlignConsecutiveAssignments``, ``AlignConsecutiveBitFields``,
  ``AlignConsecutiveDeclarations`` and ``AlignConsecutiveMacros`` have been modified to allow
  alignment across empty lines and/or comments.

- Support for Whitesmiths has been improved, with fixes for ``namespace`` blocks
  and ``case`` blocks and labels.

libclang
--------

- ...

Static Analyzer
---------------

.. 3ff220de9009 [analyzer][StdLibraryFunctionsChecker] Add POSIX networking functions
.. ...And a million other patches.
- Improve the analyzer's understanding of several POSIX functions.

.. https://reviews.llvm.org/D86533#2238207
- Greatly improved the analyzer’s constraint solver by better understanding
  when constraints are imposed on multiple symbolic values that are known to be
  equal or known to be non-equal. It will now also efficiently reject impossible
  if-branches between known comparison expressions. (Incorrectly stated as a
  11.0.0 feature in the previous release notes)

.. 820e8d8656ec [Analyzer][WebKit] UncountedLambdaCaptureChecker
- New checker: :ref:`webkit.UncountedLambdaCapturesChecker<webkit-UncountedLambdaCapturesChecker>`
  is a WebKit coding convention checker that flags raw pointers to
  reference-counted objects captured by lambdas and suggests using intrusive
  reference-counting smart pointers instead.

.. 8a64689e264c [Analyzer][WebKit] UncountedLocalVarsChecker
- New checker: :ref:`alpha.webkit.UncountedLocalVarsChecker<alpha-webkit-UncountedLocalVarsChecker>`
  is a WebKit coding convention checker that intends to make sure that any
  uncounted local variable is backed by a ref-counted object with lifetime that
  is strictly larger than the scope of the uncounted local variable.

.. i914f6c4ff8a4 [StaticAnalyzer] Support struct annotations in FuchsiaHandleChecker
- ``fuchia.HandleChecker`` now recognizes handles in structs; All the handles
  referenced by the structure (direct value or ptr) would be treated as
  containing the release/use/acquire annotations directly.

.. 8deaec122ec6 [analyzer] Update Fuchsia checker to catch releasing unowned handles.
- Fuchsia checkers can detect the release of an unowned handle.

- Numerous fixes and improvements to bug report generation.

.. _release-notes-ubsan:

Undefined Behavior Sanitizer (UBSan)
------------------------------------

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
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us via the `mailing
list <https://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
