========================================
Clang 13.0.0 Release Notes
========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 13.0.0. Here we
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

What's New in Clang 13.0.0?
===========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Guaranteed tail calls are now supported with statement attributes
  ``[[clang::musttail]]`` in C++ and ``__attribute__((musttail))`` in C. The
  attribute is applied to a return statement (not a function declaration),
  and an error is emitted if a tail call cannot be guaranteed, for example if
  the function signatures of caller and callee are not compatible. Guaranteed
  tail calls enable a class of algorithms that would otherwise use an
  arbitrary amount of stack space.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ...

Non-comprehensive list of changes in this release
-------------------------------------------------

- The default value of _MSC_VER was raised from 1911 to 1914. MSVC 19.14 has the
  support to overaligned objects on x86_32 which is required for some LLVM
  passes.

New Compiler Flags
------------------

- ``-Wreserved-identifier`` emits warning when user code uses reserved
  identifiers.

- ``Wunused-but-set-parameter`` and ``-Wunused-but-set-variable`` emit warnings
  when a parameter or a variable is set but not used.

- ``-fstack-usage`` generates an extra .su file per input source file. The .su
  file contains frame size information for each function defined in the source
  file.

- ``-Wnull-pointer-subtraction`` emits warning when user code may have
  undefined behaviour due to subtraction involving a null pointer.

Deprecated Compiler Flags
-------------------------

- ...

Modified Compiler Flags
-----------------------

- -Wshadow now also checks for shadowed structured bindings
- ``-B <prefix>`` (when ``<prefix>`` is a directory) was overloaded to additionally
  detect GCC installations under ``<prefix>`` (``lib{,32,64}/gcc{,-cross}/$triple``).
  This behavior was incompatible with GCC, caused interop issues with
  ``--gcc-toolchain``, and was thus dropped. Specify ``--gcc-toolchain=<dir>``
  instead. ``-B``'s other GCC-compatible semantics are preserved:
  ``$prefix/$triple-$file`` and ``$prefix$file`` are searched for executables,
  libraries, includes, and data files used by the compiler.
- ``-Wextra`` now also implies ``-Wnull-pointer-subtraction.``

Removed Compiler Flags
-------------------------

- The clang-cl ``/fallback`` flag, which made clang-cl invoke Microsoft Visual
  C++ on files it couldn't compile itself, has been removed.

- ``-Wreturn-std-move-in-c++11``, which checked whether an entity is affected by
  `CWG1579 <https://wg21.link/CWG1579>`_ to become implicitly movable, has been
  removed.

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- ...

- Added support for C++11-style ``[[]]`` attributes on using-declarations, as a
  clang extension.

Windows Support
---------------

- Fixed reading ``long double`` arguments with ``va_arg`` on x86_64 MinGW
  targets.

C Language Changes in Clang
---------------------------

- ...

C++ Language Changes in Clang
-----------------------------

- The oldest supported GNU libstdc++ is now 4.8.3 (released 2014-05-22).
  Clang workarounds for bugs in earlier versions have been removed.

- ...

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^
...

C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^
...

Objective-C Language Changes in Clang
-------------------------------------

OpenCL Kernel Language Changes in Clang
---------------------------------------


Command-line interface changes:

- All builtin types, macros and function declarations are now added by default
  without any command-line flags. A flag is provided ``-cl-no-stdinc`` to
  suppress the default declarations non-native to the compiler.

- Clang now compiles using OpenCL C version 1.2 by default if no version is
  specified explicitly from the command line.

- Clang now supports ``.clcpp`` file extension for sources written in
  C++ for OpenCL.

- Clang now accepts ``-cl-std=clc++1.0`` that sets C++ for OpenCL to
  the version 1.0 explicitly.

Misc common changes:

- Added ``NULL`` definition in internal headers for standards prior to the
  version 2.0.

- Simplified use of pragma in extensions for ``double``, images, atomics,
  subgroups, Arm dot product extension. There are less cases where extension
  pragma is now required by clang to compile kernel sources.

- Added missing ``as_size``/``as_ptrdiff``/``as_intptr``/``as_uintptr_t``
  operators to internal headers.

- Added new builtin function for ndrange, ``cl_khr_subgroup_extended_types``,
  ``cl_khr_subgroup_non_uniform_vote``, ``cl_khr_subgroup_ballot``,
  ``cl_khr_subgroup_non_uniform_arithmetic``, ``cl_khr_subgroup_shuffle``,
  ``cl_khr_subgroup_shuffle_relative``, ``cl_khr_subgroup_clustered_reduce``
  into the default Tablegen-based header.

- Added online documentation for Tablegen-based header, OpenCL 3.0 support,
  new clang extensions.

- Fixed OpenCL C language version and SPIR address space reporting in DWARF.

New extensions:

- ``cl_khr_integer_dot_product`` for dedicated support of dot product.

- ``cl_khr_extended_bit_ops`` for dedicated support of extra binary operations.

- ``__cl_clang_bitfields`` for use of bit-fields in the kernel code.

- ``__cl_clang_non_portable_kernel_param_types`` for relaxing some restrictions
  to types of kernel parameters.

OpenCL C 3.0 related changes:

- Added parsing support for the optionality of generic address space, images 
  (including 3d writes and ``read_write`` access qualifier), pipes, program
  scope variables, double-precision floating-point support. 

- Added optionality support for builtin functions (in ``opencl-c.h`` header)
  for generic address space, C11 atomics.  

- Added ``memory_scope_all_devices`` enum for the atomics in internal headers.

- Enabled use of ``.rgba`` vector components.

C++ for OpenCL related changes:

- Added ``__remove_address_space`` metaprogramming utility in internal headers
  to allow removing address spaces from types.

- Improved overloads resolution logic for constructors wrt address spaces.

- Improved diagnostics of OpenCL specific types and address space qualified
  types in ``reinterpret_cast`` and template functions.

- Fixed ``NULL`` macro in internal headers to be compatible with C++.

- Fixed use of ``half`` type.

ABI Changes in Clang
--------------------

OpenMP Support in Clang
-----------------------

- Support for loop transformation directives from OpenMP 5.1 have been added.
  ``#pragma omp unroll`` is a standardized alternative to ``#pragma unroll``
  (or ``#pragma clang loop unroll(enable)``) but also allows composition with
  other OpenMP loop associated constructs as in

  .. code-block:: c

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < n; ++i)

  ``#pragma omp tile`` applies tiling to a perfect loop nest using a
  user-defined tile size.

  .. code-block:: c

    #pragma omp tile sizes(8,8)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)

- ...

CUDA Support in Clang
---------------------

- ...

X86 Support in Clang
--------------------

- ...

Internal API Changes
--------------------

These are major API changes that have happened since the 12.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

- ...

Build System Changes
--------------------

These are major changes to the build system that have happened since the 12.0.0
release of Clang. Users of the build system should adjust accordingly.

- The option ``LIBCLANG_INCLUDE_CLANG_TOOLS_EXTRA`` no longer exists. There were
  two releases with that flag forced off, and no uses were added that forced it
  on. The recommended replacement is clangd.

- ...

AST Matchers
------------

- ...

clang-format
------------

- Option ``SpacesInLineCommentPrefix`` has been added to control the
  number of spaces in a line comments prefix.

- Option ``SortIncludes`` has been updated from a ``bool`` to an
  ``enum`` with backwards compatibility. In addition to the previous
  ``true``/``false`` states (now ``CaseSensitive``/``Never``), a third
  state has been added (``CaseInsensitive``) which causes an alphabetical sort
  with case used as a tie-breaker.

  .. code-block:: c++

    // Never (previously false)
    #include "B/A.h"
    #include "A/B.h"
    #include "a/b.h"
    #include "A/b.h"
    #include "B/a.h"

    // CaseSensitive (previously true)
    #include "A/B.h"
    #include "A/b.h"
    #include "B/A.h"
    #include "B/a.h"
    #include "a/b.h"

    // CaseInsensitive
    #include "A/B.h"
    #include "A/b.h"
    #include "a/b.h"
    #include "B/A.h"
    #include "B/a.h"

- ``BasedOnStyle: InheritParentConfig`` allows to use the ``.clang-format`` of
  the parent directories to overwrite only parts of it.

- Option ``IndentAccessModifiers`` has been added to be able to give access
  modifiers their own indentation level inside records.

- Option ``PPIndentWidth`` has been added to be able to configure pre-processor
  indentation independent from regular code.

- Option ``ShortNamespaceLines`` has been added to give better control
  over ``FixNamespaceComments`` when determining a namespace length.

- Support for Whitesmiths has been improved, with fixes for ``namespace`` blocks
  and ``case`` blocks and labels.

- Option ``EmptyLineAfterAccessModifier`` has been added to remove, force or keep
  new lines after access modifiers.

- Checks for newlines in option ``EmptyLineBeforeAccessModifier`` are now based
  on the formatted new lines and not on the new lines in the file. (Fixes
  https://llvm.org/PR41870.)

- Option ``SpacesInAngles`` has been improved, it now accepts ``Leave`` value
  that allows to keep spaces where they are already present.

- Option ``AllowShortIfStatementsOnASingleLine`` has been improved, it now
  accepts ``AllIfsAndElse`` value that allows to put "else if" and "else" short
  statements on a single line. (Fixes https://llvm.org/PR50019.)

- Option ``BreakInheritanceList`` gets a new style, ``AfterComma``. It breaks
  only after the commas that separate the base-specifiers.

- Option ``LambdaBodyIndentation`` has been added to control how the body of a
  lambda is indented. The default ``Signature`` value indents the body one level
  relative to whatever indentation the signature has. ``OuterScope`` lets you
  change that so that the lambda body is indented one level relative to the scope
  containing the lambda, regardless of where the lambda signature was placed.

- Option ``IfMacros`` has been added. This lets you define macros that get
  formatted like conditionals much like ``ForEachMacros`` get styled like
  foreach loops.

- ``git-clang-format`` no longer formats changes to symbolic links. (Fixes
  https://llvm.org/PR46992.)

- Makes ``PointerAligment: Right`` working with ``AlignConsecutiveDeclarations``.
  (Fixes https://llvm.org/PR27353)

- Option ``AlignArrayOfStructure`` has been added to allow for ordering array-like
  initializers.

- Support for formatting JSON file (\*.json) has been added to clang-format.

libclang
--------

- Make libclang SONAME independent from LLVM version. It will be updated only when
  needed. Defined in CLANG_SONAME (clang/tools/libclang/CMakeLists.txt).
  `More details <https://lists.llvm.org/pipermail/cfe-dev/2021-June/068423.html>`_

Static Analyzer
---------------

.. 2407eb08a574 [analyzer] Update static analyzer to be support sarif-html

- Add a new analyzer output type, ``sarif-html``, that outputs both HTML and
  Sarif files.

.. 90377308de6c [analyzer] Support allocClassWithName in OSObjectCStyleCast checker

- Add support for ``allocClassWithName`` in OSObjectCStyleCast checker.

.. cad9b7f708e2b2d19d7890494980c5e427d6d4ea: Print time taken to analyze each function

- The option ``-analyzer-display-progress`` now also outputs analysis time for
  each function.

.. 9e02f58780ab8734e5d27a0138bd477d18ae64a1 [analyzer] Highlight arrows for currently selected event

- For bug reports in HTML format, arrows are now highlighted for the currently
  selected event.

.. Deep Majumder's GSoC'21
.. 80068ca6232b [analyzer] Fix for faulty namespace test in SmartPtrModelling
.. d825309352b4 [analyzer] Handle std::make_unique
.. 0cd98bef1b6f [analyzer] Handle std::swap for std::unique_ptr
.. 13fe78212fe7 [analyzer] Handle << operator for std::unique_ptr
.. 48688257c52d [analyzer] Model comparision methods of std::unique_ptr
.. f8d3f47e1fd0 [analyzer] Updated comments to reflect D85817
.. 21daada95079 [analyzer] Fix static_cast on pointer-to-member handling

- While still in alpha, ``alpha.cplusplus.SmartPtr`` received numerous
  improvements and nears production quality.

.. 21daada95079 [analyzer] Fix static_cast on pointer-to-member handling
.. 170c67d5b8cc [analyzer] Use the MacroExpansionContext for macro expansions in plists
.. 02b51e5316cd [analyzer][solver] Redesign constraint ranges data structure
.. 3085bda2b348 [analyzer][solver] Fix infeasible constraints (PR49642)
.. 015c39882ebc [Analyzer] Infer 0 value when the divisible is 0 (bug fix)
.. 90377308de6c [analyzer] Support allocClassWithName in OSObjectCStyleCast checker
.. df64f471d1e2 [analyzer] DynamicSize: Store the dynamic size
.. e273918038a7 [analyzer] Track leaking object through stores
.. 61ae2db2d7a9 [analyzer] Adjust the reported variable name in retain count checker
.. 50f17e9d3139 [analyzer] RetainCountChecker: Disable reference counting for OSMetaClass.

- Various fixes and improvements, including modeling of casts (such as 
  ``std::bit_cast<>``), constraint solving, explaining bug-causing variable
  values, macro expansion notes, modeling the size of dynamic objects and the
  modeling and reporting of Objective C/C++ retain count related bugs. These
  should reduce false positives and make the remaining reports more readable.

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
