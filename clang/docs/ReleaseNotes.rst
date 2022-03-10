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
     `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C/OpenCL
frontend, part of the LLVM Compiler Infrastructure, release |release|. Here we
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

What's New in Clang |release|?
==============================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Added SPIR-V triple and binary generation using external ``llvm-spirv`` tool.
  For more details refer to :ref:`the SPIR-V support section <spir-v>`.
- Completed support of OpenCL C 3.0 and C++ for OpenCL 2021 at experimental
  state.

- Prebuilt AIX7.2 TL5 SP3+ binary available with following notes and
  limitations:

  - C++ driver modes use the system libc++ headers. These headers are included
    in the optional ``libc++.adt.include`` fileset on AIX.
  - LTO, although not disabled, is not meaningfully functional for practical
    use.
  - Shared libraries builds (``-shared``) must use explicit symbol export
    options and/or export lists (e.g., with ``-bE:``) on the link step. Clang
    currently will not automatically generate symbol export lists as implicit
    linker inputs.

- ``float.h`` now exposes (in hosted mode) extensions made available from the
  AIX system header.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- -Wbitwise-instead-of-logical (part of -Wbool-operation) warns about use of bitwise operators with boolean operands which have side effects.

- Added diagnostic groups to control diagnostics for attribute extensions by
  adding groups ``-Wc++N-attribute-extensions`` (where ``N`` is the standard
  release being diagnosed against). These new groups are automatically implied
  when passing ``-Wc++N-extensions``. Resolves PR33518.

- Support ``-Wdeclaration-after-statement`` with C99 and later standards, and
  not just C89, matching GCC's behaviour. A notable usecase is supporting style
  guides that forbid mixing declarations and code, but want to move to newer C
  standards.

Non-comprehensive list of changes in this release
-------------------------------------------------

- Maximum _ExtInt size was decreased from 16,777,215 bits to 8,388,608 bits.
  Motivation for this was discussed in PR51829.
- Configuration file syntax extended with ``<CFGDIR>`` token. This expands to
  the base path of the current config file. See :ref:`configuration-files` for
  details.
- The ``-E -P`` preprocessor output now always omits blank lines, matching
  gcc behaviour. Previously, up to 8 consecutive blank lines could appear
  in the output.
- AIX platform-related predefined macros added:
  ``_ARCH_PPC64``, ``__HOS_AIX__``, ``__PPC``, ``__THW_BIG_ENDIAN__``,
  ``__THW_PPC__``, and ``__powerpc``

New Compiler Flags
------------------

- Clang plugin arguments can now be passed through the compiler driver via
  ``-fplugin-arg-pluginname-arg``, similar to GCC's ``-fplugin-arg``.
- The ``-mno-bti-at-return-twice`` flag will make sure a BTI instruction won't
  be added after a setjmp or possible other return-twice construct (ARM backend
  only).
- The ``--start-no-unused-arguments`` and ``--end-no-unused-arguments`` flags
  allow silencing warnings about unused arguments for only a subset of
  the command line arguments, keeping potential warnings for other arguments
  outside of such a region.
- ``-falign-loops=N`` (N is a power of 2) is now supported for non-LTO cases.
  (`D106701 <https://reviews.llvm.org/D106701>`_)
- The ``-fminimize-whitespace`` flag allows removing redundant whitespace
  from preprocessor output (``-E`` flag). When combined with ``-P``, this
  includes newlines. Otherwise, only indention is removed (other horizontal
  whitespace is always collapsed).
  The motivation is to improve compiler cache hit rate by becoming invariant
  to whitespace changes, such as reformatting using clang-format. Patches
  for `ccache <https://github.com/ccache/ccache/pull/815>`_ and
  `sccache <https://github.com/mozilla/sccache/pull/1055>`_ are under review.

- Clang now accepts "allowlist" spelling for ``-objcmt-allowlist-dir-path``.

Deprecated Compiler Flags
-------------------------

- -Wweak-template-vtables has been deprecated and no longer has any effect. The
  flag will be removed in the next release.

Modified Compiler Flags
-----------------------

- Support has been added for the following processors (``-mcpu`` identifiers in parentheses):

  - RISC-V SiFive E20 (``sifive-e20``).
  - RISC-V SiFive E21 (``sifive-e21``).
  - RISC-V SiFive E24 (``sifive-e24``).
  - RISC-V SiFive E34 (``sifive-e34``).
  - RISC-V SiFive S21 (``sifive-s21``).
  - RISC-V SiFive S51 (``sifive-s51``).
  - RISC-V SiFive S54 (``sifive-s54``).
  - RISC-V SiFive S76 (``sifive-s76``).
  - Arm Cortex-X1C (``cortex-x1c``)
  - Arm Cortex-X2 (``cortex-x2``)
  - Arm Cortex-A510 (``cortex-a510``)
  - Arm Cortex-A710 (``cortex-a710``)

- Support has been added for the following architectures (``-march`` identifiers in parentheses):

  - Armv9-A (``armv9-a``).
  - Armv9.1-A (``armv9.1-a``).
  - Armv9.2-A (``armv9.2-a``).

- ``-r`` now implies ``-nostdlib`` for many toolchains, matching GCC.
  (`D116843 <https://reviews.llvm.org/D116843>`_)

Removed Compiler Flags
-------------------------

- The legacy ``-gz=zlib-gnu`` and ``-Wa,--compress-debug-sections=zlib-gnu``
  have been removed.
- ``-fno-experimental-new-pass-manager`` has been removed.
  ``-flegacy-pass-manager`` can be used as a makeshift,
  Using the legacy pass manager for the optimization pipeline was deprecated in
  13.0.0 and will be removed after 14.0.0.

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- Attributes loaded as clang plugins which are sensitive to LangOpts must
  now override ``acceptsLangOpts`` instead of ``diagLangOpts``.
  Returning false will produce a generic "attribute ignored" diagnostic, as
  with clang's built-in attributes.
  If plugins want to provide richer diagnostics, they can do so when the
  attribute is handled instead, e.g. in ``handleDeclAttribute``.
  (This was changed in order to better support attributes in code completion).

- __has_cpp_attribute, __has_c_attribute, __has_attribute, and __has_declspec
  will now macro expand their argument. This causes a change in behavior for
  code using ``__has_cpp_attribute(__clang__::attr)`` (and same for
  ``__has_c_attribute``) where it would previously expand to ``0`` for all
  attributes, but will now issue an error due to the expansion of the
  predefined ``__clang__`` macro.

- Improved handling of ``__attribute__((__aligned__))`` on AIX to match GCC.

Windows Support
---------------

- An MSVC compatibility workaround for C++ operator names was removed. As a
  result, the ``<query.h>`` Windows SDK header may not compile out of the box.
  Users should use a recent SDK and pass ``-DQUERY_H_RESTRICTION_PERMISSIVE``
  or pass ``/permissive`` to disable C++ operator names altogether. See
  `PR42427 <https://llvm.org/pr42427>` for more info.

- Add support for MSVC-compatible ``/hotpatch`` flag in clang-cl, and equivalent
  -cc1 flag ``-fms-hotpatch``. Along with the linker flag ``/functionpadmin``
  this creates executable images suitable for runtime code patching. This flag
  is only required for x86/x64 targets; ARM/ARM64 simply needs the linker
  ``/functionpadmin``.

  With this addition, clang-cl can be used in live code patching scenarios,
  along with tools such as Live++ or Recode. Microsoft Edit and Continue isn't
  currently supported.

- Support for on-demand initialization of TLS variables was added.

- Improved code generation for ARM, by assuming less strict alignment
  requirements for instructions (just like other OSes do).

- Fixed using the ``-m32`` flag in x86_64 MinGW setups, by e.g. making ``-m32``
  pick i686 instead of i386, if there is no i386 sysroot, but only one for
  i686.

- Fixed passing the ``--no-demangle`` option through to the linker for MinGW
  targets.

C Language Changes in Clang
---------------------------

- The value of ``__STDC_VERSION__`` has been bumped to ``202000L`` when passing
  ``-std=c2x`` so that it can be distinguished from C17 mode. This value is
  expected to change again when C23 is published.
- Wide multi-characters literals such as ``L'ab'`` that would previously be interpreted as ``L'b'``
  are now ill-formed in all language modes. The motivation for this change is outlined in
  `P2362 <wg21.link/P2362>`_.
- Support for ``__attribute__((error("")))`` and
  ``__attribute__((warning("")))`` function attributes have been added.
- The maximum allowed alignment has been increased from 2^29 to 2^32.
- Clang now supports the ``_BitInt(N)`` family of bit-precise integer types
  from C23. This type was previously exposed as ``_ExtInt(N)``, which is now a
  deprecated alias for ``_BitInt(N)`` (so diagnostics will mention ``_BitInt``
  even if source uses ``_ExtInt``). ``_BitInt(N)`` and ``_ExtInt(N)`` are the
  same types in all respects beyond spelling and the deprecation warning.
  ``_BitInt(N)`` is supported as an extension in older C modes and in all C++
  modes. Note: the ABI for ``_BitInt(N)`` is still in the process of being
  stabilized, so this type should not yet be used in interfaces that require
  ABI stability. The maximum width supported by Clang can be obtained from the
  ``BITINT_MAXWIDTH`` macro in ``<limits.h>``. Currently, Clang supports bit
  widths <= 128 because backends are not yet able to cope with some math
  operations (like division) on wider integer types. See
  `PR44994 <https://github.com/llvm/llvm-project/issues/44994>`_ for more
  information.
- When using ``asm goto`` with outputs whose constraint modifier is ``"+"``, we
  now change the numbering of the labels to occur after hidden tied inputs for
  better compatibility with GCC.  For better portability between different
  compilers and versions, symbolic references rather than numbered references
  should be preferred. See
  `this thread <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=103640>` for more
  info.

- Implemented `WG14 N2412 <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2412.pdf>`_,
  which adds ``*_WIDTH`` macros to limits.h and stdint.h to report the bit
  width of various integer datatypes.

- The ``ATOMIC_VAR_INIT`` macro from ``<stdatomic.h>`` is now diagnosed as
  deprecated in C17 and later. The diagnostic can be disabled by defining the
  ``_CLANG_DISABLE_CRT_DEPRECATION_WARNINGS`` macro prior to including the
  header.

C++ Language Changes in Clang
-----------------------------

- ...

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

- The ``ATOMIC_VAR_INIT`` and ``ATOMIC_FLAG_INIT`` macros from the C standard
  library ``<stdatomic.h>`` header are now diagnosed as deprecated in C++20 and
  later. Note, the behavior is specific to the inclusion of ``<stdatomic.h>``
  in C++ code; the STL ``<atomic>`` header also controls the behavior of these
  macros and is not affected by these changes. The ``<stdatomic.h>`` diagnostic
  can be disabled by defining the ``_CLANG_DISABLE_CRT_DEPRECATION_WARNINGS``
  macro prior to including the header.


C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Implemented `P1938R3: if consteval <https://wg21.link/P1938R3>`_.
- Implemented `P2360R0: Extend init-statement to allow alias-declaration <https://wg21.link/P2360R0>`_.


CUDA Language Changes in Clang
------------------------------

- Clang now supports CUDA versions up to 11.5.
- Default GPU architecture has been changed from sm_20 to sm_35.

Objective-C Language Changes in Clang
-------------------------------------

OpenCL Kernel Language Changes in Clang
---------------------------------------

OpenCL 3.0 is completed, but remains experimental:

- Added parsing support for optionality of device-side enqueue and blocks.
- Added missing support for optionality of various builtin functions:

  - ``read_write`` images, pipes, collective workgroup in the default header.
  - ``read_write`` images, named address space atomics in internal ``opencl-c.h``
    (enabled via ``-finclude-default-header`` frontend flag).

C++ for OpenCL:

- Added experimental support of C++ for OpenCL 2021 as per `the provisional
  language documentation
  <https://github.com/KhronosGroup/OpenCL-Docs/releases/tag/cxxforopencl-docrev2021.12>`_.
  Support of all optional features is aligned with OpenCL 3.0.
- Added ``__remove_address_space`` utility (documentation available in
  :doc:`LanguageExtensions`).
- Fixed address space for temporaries (to be ``__private``).
- Disallowed static kernel functions.
- Fixed implicit definition of ``__cpp_threadsafe_static_init`` macro.

Misc changes:

- Added generation of SPIR-V binaries via external ``llvm-spirv`` tool.
  For more details refer to :ref:`the SPIR-V support section <spir-v>`.
- Added new extensions for ``atomic_half`` and ``cl_ext_float_atomics``.
- Fixed/improved support of ``vload``/``vstore``.
- Fixed incorrect ``as_type`` support for 3-element vector types.

ABI Changes in Clang
--------------------

- The ``_ExtInt(N)`` extension has been standardized in C23 as ``_BitInt(N)``.
  The mangling of this type in C++ has accordingly changed: under the Microsoft
  ABI it is now mangled using the ``_BitInt`` spelling, and under the Itanium ABI
  it is now mangled using a dedicated production. Note: the ABI for ``_BitInt(N)``
  is still in the process of being stabilized, so this type should not yet be
  used in interfaces that require ABI stability.

OpenMP Support in Clang
-----------------------

- ``clang-nvlink-wrapper`` tool introduced to support linking of cubin files archived in an archive. See :doc:`ClangNvlinkWrapper`.
- ``clang-linker-wrapper`` tool introduced to support linking using a new OpenMP target offloading method. See :doc:`ClangLinkerWrapper`.

CUDA Support in Clang
---------------------

- ...

X86 Support in Clang
--------------------

- Support for ``AVX512-FP16`` instructions has been added.

DWARF Support in Clang
----------------------

- The default DWARF version has increased from DWARFv4 to DWARFv5.  You can opt
  back in to the old behavior with ``-gdwarf-4`` or ``-fdebug-default-version=4``.
  Some platforms (Darwin, Android, and SCE for instance) already opt out of this
  version bump as is suitable for the platform

Arm and AArch64 Support in Clang
--------------------------------

- The -mtune flag is no longer ignored for AArch64. It is now possible to
  tune code generation for a particular CPU with -mtune without setting any
  architectural features. For example, compiling with
  "-mcpu=generic -mtune=cortex-a57" will not enable any Cortex-A57 specific
  architecture features, but will enable certain optimizations specific to
  Cortex-A57 CPUs and enable the use of a more accurate scheduling model.

- The --aarch64-none-elf target now uses the BareMetal driver rather than the
  GNU driver. Programs that depend on clang invoking GCC as the linker driver
  should use GCC as the linker in the build system.

- The ``-mbranch-protection`` flag will now also work for the ARM backend.

- The ``attribute((target("branch-protection=...)))`` attributes will now also
  work for the ARM backend.

SPIR-V Support in Clang
-----------------------

- Added triple/target ``spirv32`` and ``spirv64`` for 32-bit and 64-bit SPIR-V
  respectively.
- Added generation of binaries via external ``llvm-spirv`` tool. This can now
  be used for HIP or OpenCL.
- Added linking of separate object files in SPIR-V format using external
  ``spirv-link`` tool.


Floating Point Support in Clang
-------------------------------
- The default setting of FP contraction (FMA) is now -ffp-contract=on (for
  languages other than CUDA/HIP) even when optimization is off. Previously,
  the default behavior was equivalent to -ffp-contract=off (-ffp-contract
  was not set).
  Related to this, the switch -ffp-model=precise now implies -ffp-contract=on
  rather than -ffp-contract=fast, and the documentation of these features has
  been clarified. Previously, the documentation claimed that -ffp-model=precise
  was the default, but this was incorrect because the precise model implied
  -ffp-contract=fast, whereas the (now corrected) default behavior is
  -ffp-contract=on.
  -ffp-model=precise is now exactly the default mode of the compiler.
- -fstrict-float-cast-overflow no longer has target specific behavior. Clang
  will saturate towards the smallest and largest representable integer values.
  NaNs will be converted to zero.

Internal API Changes
--------------------

- A new sugar ``Type`` AST node represents types accessed via a C++ using
  declaration. Given code ``using std::error_code; error_code x;``, ``x`` has
  a ``UsingType`` which desugars to the previous ``RecordType``.

Build System Changes
--------------------

- Linux distros can specify ``-DCLANG_DEFAULT_PIE_ON_LINUX=On`` to use ``-fPIE`` and
  ``-pie`` by default. This matches GCC installations on many Linux distros
  (configured with ``--enable-default-pie``).
  (`D113372 <https://reviews.llvm.org/D113372>`_)

AST Matchers
------------

- ``TypeLoc`` AST Matchers are now available. These matchers provide helpful
  utilities for matching ``TypeLoc`` nodes, such as the ``pointerTypeLoc``
  matcher or the ``hasReturnTypeLoc`` matcher. The addition of these matchers
  was made possible by changes to the handling of ``TypeLoc`` nodes that
  allows them to enjoy the same static type checking as other AST node kinds.
- ``LambdaCapture`` AST Matchers are now available. These matchers allow for
  the binding of ``LambdaCapture`` nodes. The ``LambdaCapture`` matchers added
  include the ``lambdaCapture`` node matcher, the ``capturesVar`` traversal
  matcher, and ``capturesThis`` narrowing matcher.
- The ``hasAnyCapture`` matcher now only accepts an inner matcher of type
  ``Matcher<LambdaCapture>``. The matcher originally accepted an inner matcher
  of type ``Matcher<CXXThisExpr>`` or ``Matcher<VarDecl>``.
- The ``usingType`` matcher is now available and needed to refer to types that
  are referred to via using C++ using declarations.
  The associated ``UsingShadowDecl`` can be matched using ``throughUsingDecl``
  and the underlying ``Type`` with ``hasUnderlyingType``.
  ``hasDeclaration`` continues to see through the alias and apply to the
  underlying type.
- Added the ``isConsteval`` matcher to match ``consteval`` function
  declarations as well as `if consteval` and `if ! consteval` statements.
- Added the ``isConstinit`` matcher to match ``constinit`` variable
  declarations.

clang-format
------------

- Option ``AllowShortEnumsOnASingleLine: false`` has been improved, it now
  correctly places the opening brace according to ``BraceWrapping.AfterEnum``.

- Option ``AlignAfterOpenBracket: BlockIndent`` has been added. If set, it will
  always break after an open bracket, if the parameters don't fit on a single
  line. Closing brackets will be placed on a new line.

- Option ``QualifierAlignment`` has been added in order to auto-arrange the
  positioning of specifiers/qualifiers
  `const` `volatile` `static` `inline` `constexpr` `restrict`
  in variable and parameter declarations to be either ``Right`` aligned
  or ``Left`` aligned or ``Custom`` using ``QualifierOrder``.

- Option ``QualifierOrder`` has been added to allow the order
  `const` `volatile` `static` `inline` `constexpr` `restrict`
  to be controlled relative to the `type`.

- Option ``RemoveBracesLLVM`` has been added to remove optional braces of
  control statements for the LLVM style.

- Option ``SeparateDefinitionBlocks`` has been added to insert or remove empty
  lines between definition blocks including functions, classes, structs, enums,
  and namespaces.

- Add a ``Custom`` style to ``SpaceBeforeParens``, to better configure the
  space before parentheses. The custom options can be set using
  ``SpaceBeforeParensOptions``.

- The command line argument `-style=<string>` has been extended so that a specific
  format file at location <format_file_path> can be selected. This is supported
  via the syntax: `-style=file:<format_file_path>`.

- Improved C++20 Modules and Coroutines support.

- Option ``AfterOverloadedOperator`` has been added in ``SpaceBeforeParensOptions``
  to allow space between overloaded operator and opening parentheses.

libclang
--------

- ...

Static Analyzer
---------------

- ...

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
