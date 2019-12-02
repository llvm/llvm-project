========================================
Clang 10.0.0 (In-Progress) Release Notes
========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 10 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 10.0.0. Here we
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

What's New in Clang 10.0.0?
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

- -Wtautological-overlap-compare will warn on negative numbers and non-int
  types.
- -Wtautological-compare for self comparisons and
  -Wtautological-overlap-compare will now look through member and array
  access to determine if two operand expressions are the same.
- -Wtautological-bitwise-compare is a new warning group.  This group has the
  current warning which diagnoses the tautological comparison of a bitwise
  operation and a constant. The group also has the new warning which diagnoses
  when a bitwise-or with a non-negative value is converted to a bool, since
  that bool will always be true.
- -Wbitwise-conditional-parentheses will warn on operator precedence issues
  when mixing bitwise-and (&) and bitwise-or (|) operator with the
  conditional operator (?:).

Non-comprehensive list of changes in this release
-------------------------------------------------

* In both C and C++ (C17 ``6.5.6p8``, C++ ``[expr.add]``), pointer arithmetic is
  only permitted within arrays. In particular, the behavior of a program is not
  defined if it adds a non-zero offset (or in C, any offset) to a null pointer,
  or if it forms a null pointer by subtracting an integer from a non-null
  pointer, and the LLVM optimizer now uses those guarantees for transformations.
  This may lead to unintended behavior in code that performs these operations.
  The Undefined Behavior Sanitizer ``-fsanitize=pointer-overflow`` check has
  been extended to detect these cases, so that code relying on them can be
  detected and fixed.

* The Implicit Conversion Sanitizer (``-fsanitize=implicit-conversion``) has
  learned to sanitize pre/post increment/decrement of types with bit width
  smaller than ``int``.

- For X86 target, -march=skylake-avx512, -march=icelake-client,
  -march=icelake-server, -march=cascadelake, -march=cooperlake will default to
  not using 512-bit zmm registers in vectorized code unless 512-bit intrinsics
  are used in the source code. 512-bit operations are known to cause the CPUs
  to run at a lower frequency which can impact performance. This behavior can be
  changed by passing -mprefer-vector-width=512 on the command line.

New Compiler Flags
------------------

- The -fgnuc-version= flag now controls the value of ``__GNUC__`` and related
  macros. This flag does not enable or disable any GCC extensions implemented in
  Clang. Setting the version to zero causes Clang to leave ``__GNUC__`` and
  other GNU-namespaced macros, such as ``__GXX_WEAK__``, undefined.

- vzeroupper insertion on X86 targets can now be disabled with -mno-vzeroupper.
  You can also force vzeroupper insertion to be used on CPUs that normally
  wouldn't with -mvzeroupper.

Deprecated Compiler Flags
-------------------------

The following options are deprecated and ignored. They will be removed in
future versions of Clang.

- -mmpx used to enable the __MPX__ preprocessor define for the Intel MPX
  instructions. There were no MPX intrinsics.
- -mno-mpx used to disable -mmpx and is the default behavior.

- ...

Modified Compiler Flags
-----------------------

- RISC-V now sets the architecture (riscv32/riscv64) based on the value provided
  to the ``-march`` flag, overriding the target provided by ``-triple``.

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- ...

Windows Support
---------------

- Previous Clang versions contained a work-around to avoid an issue with the
  standard library headers in Visual Studio 2019 versions prior to 16.3. This
  work-around has now been removed, and users of Visual Studio 2019 are
  encouraged to upgrade to 16.3 or later, otherwise they may see link errors as
  below:

  .. code-block:: console

    error LNK2005: "bool const std::_Is_integral<int>" (??$_Is_integral@H@std@@3_NB) already defined

C Language Changes in Clang
---------------------------

- ...

C11 Feature Support
^^^^^^^^^^^^^^^^^^^

...

C++ Language Changes in Clang
-----------------------------

- The behaviour of the `gnu_inline` attribute now matches GCC, for cases
  where used without the `extern` keyword. As this is a change compared to
  how it behaved in previous Clang versions, a warning is emitted for this
  combination.

C++1z Feature Support
^^^^^^^^^^^^^^^^^^^^^

...

Objective-C Language Changes in Clang
-------------------------------------

- In both Objective-C and
  Objective-C++, ``-Wcompare-distinct-pointer-types`` will now warn when
  comparing ObjC ``Class`` with an ObjC instance type pointer.

  .. code-block:: objc

    Class clz = ...;
    MyType *instance = ...;
    bool eq = (clz == instance); // Previously undiagnosed, now warns.

- Objective-C++ now diagnoses conversions between ``Class`` and ObjC
  instance type pointers. Such conversions already emitted an
  on-by-default ``-Wincompatible-pointer-types`` warning in Objective-C
  mode, but had inadvertently been missed entirely in
  Objective-C++. This has been fixed, and they are now diagnosed as
  errors, consistent with the usual C++ treatment for conversions
  between unrelated pointer types.

  .. code-block:: objc

    Class clz = ...;
    MyType *instance = ...;
    clz = instance; // Previously undiagnosed, now an error.
    instance = clz; // Previously undiagnosed, now an error.

  One particular issue you may run into is attempting to use a class
  as a key in a dictionary literal. This will now result in an error,
  because ``Class`` is not convertable to ``id<NSCopying>``. (Note that
  this was already a warning in Objective-C mode.) While an arbitrary
  ``Class`` object is not guaranteed to implement ``NSCopying``, the
  default metaclass implementation does. Therefore, the recommended
  solution is to insert an explicit cast to ``id``, which disables the
  type-checking here.

 .. code-block:: objc

    Class cls = ...;

    // Error: cannot convert from Class to id<NSCoding>.
    NSDictionary* d = @{cls : @"Hello"};

    // Fix: add an explicit cast to 'id'.
    NSDictionary* d = @{(id)cls : @"Hello"};

OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------

- gcc passes vectors of __int128 in memory on X86-64. Clang historically
  broke the vectors into multiple scalars using two 64-bit values for each
  element. Clang now matches the gcc behavior on Linux and NetBSD. You can
  switch back to old API behavior with flag: -fclang-abi-compat=9.0.

- RISC-V now chooses a default ``-march=`` and ``-mabi=`` to match (in almost
  all cases) the GCC defaults. On baremetal targets, where neither ``-march=``
  nor ``-mabi=`` are specified, Clang now differs from GCC by defaulting to
  ``-march=rv32imac -mabi=ilp32`` or ``-march=rv64imac -mabi=lp64`` depending on
  the architecture in the target triple. These do not always match the defaults
  in Clang 9. We strongly suggest that you explicitly pass `-march=` and
  `-mabi=` when compiling for RISC-V, due to how extensible this architecture
  is.

OpenMP Support in Clang
-----------------------

- ...

CUDA Support in Clang
---------------------

- ...

Internal API Changes
--------------------

These are major API changes that have happened since the 9.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

- libTooling APIs that transfer ownership of `FrontendAction` objects now pass
  them by `unique_ptr`, making the ownership transfer obvious in the type
  system. `FrontendActionFactory::create()` now returns a
  `unique_ptr<FrontendAction>`. `runToolOnCode`, `runToolOnCodeWithArgs`,
  `ToolInvocation::ToolInvocation()` now take a `unique_ptr<FrontendAction>`.

Build System Changes
--------------------

These are major changes to the build system that have happened since the 9.0.0
release of Clang. Users of the build system should adjust accordingly.

- In 8.0.0 and below, the install-clang-headers target would install clang's
  resource directory headers. This installation is now performed by the
  install-clang-resource-headers target. Users of the old install-clang-headers
  target should switch to the new install-clang-resource-headers target. The
  install-clang-headers target now installs clang's API headers (corresponding
  to its libraries), which is consistent with the install-llvm-headers target.

- In 9.0.0 and later Clang added a new target, clang-cpp, which generates a
  shared library comprised of all the clang component libraries and exporting
  the clang C++ APIs. Additionally the build system gained the new
  "CLANG_LINK_CLANG_DYLIB" option, which defaults Off, and when set to On, will
  force clang (and clang-based tools) to link the clang-cpp library instead of
  statically linking clang's components. This option will reduce the size of
  binary distributions at the expense of compiler performance.

- ...

AST Matchers
------------

- ...

clang-format
------------

- The ``Standard`` style option specifies which version of C++ should be used
  when parsing and formatting C++ code. The set of allowed values has changed:

  - ``Latest`` will always enable new C++ language features.
  - ``c++03``, ``c++11``, ``c++14``, ``c++17``, ``c++20`` will pin to exactly
    that language version.
  - ``Auto`` is the default and detects style from the code (this is unchanged).

  The previous values of ``Cpp03`` and ``Cpp11`` are deprecated. Note that
  ``Cpp11`` is treated as ``Latest``, as this was always clang-format's
  behavior. (One motivation for this change is the new name describes the
  behavior better).

- Clang-format has a new option called ``--dry-run`` or ``-n`` to emit a
  warning for clang-format violations. This can be used together
  with --ferror-limit=N to limit the number of warnings per file and --Werror
  to make warnings into errors.

- Option *IncludeIsMainSourceRegex* has been added to allow for additional
  suffixes and file extensions to be considered as a source file
  for execution of logic that looks for "main *include* file" to put
  it on top.

  By default, clang-format considers *source* files as "main" only when
  they end with: ``.c``, ``.cc``, ``.cpp``, ``.c++``, ``.cxx``,
  ``.m`` or ``.mm`` extensions. This config option allows to
  extend this set of source files considered as "main".

  For example, if this option is configured to ``(Impl\.hpp)$``,
  then a file ``ClassImpl.hpp`` is considered "main" (in addition to
  ``Class.c``, ``Class.cc``, ``Class.cpp`` and so on) and "main
  include file" logic will be executed (with *IncludeIsMainRegex* setting
  also being respected in later phase). Without this option set,
  ``ClassImpl.hpp`` would not have the main include file put on top
  before any other include.

- Options ``DeriveLineEnding`` and  ``UseCRLF`` have been added to allow
  clang-format to control the newlines. ``DeriveLineEnding`` is by default
  ``true`` and reflects is the existing mechanism, which based is on majority
  rule. The new options allows this to be turned off and ``UseCRLF`` to control
  the decision as to which sort of line ending to use.

- Option ``SpaceBeforeSquareBrackets`` has been added to insert a space before
  array declarations.

  .. code-block:: c++

    int a [5];    vs    int a[5];

- Clang-format now supports JavaScript null operators.

  .. code-block:: c++

    const x = foo ?? default;
    const z = foo?.bar?.baz;

libclang
--------

- ...

Static Analyzer
---------------

- The Clang analyzer checker ``DeadStores`` gets a new option called
  ``WarnForDeadNestedAssignments`` to detect nested dead assignments
  (enabled by default).
- ...

.. _release-notes-ubsan:

Undefined Behavior Sanitizer (UBSan)
------------------------------------

- * The ``pointer-overflow`` check was extended added to catch the cases where
    a non-zero offset is applied to a null pointer, or the result of
    applying the offset is a null pointer.

    .. code-block:: c++

      #include <cstdint> // for intptr_t

      static char *getelementpointer_inbounds(char *base, unsigned long offset) {
        // Potentially UB.
        return base + offset;
      }

      char *getelementpointer_unsafe(char *base, unsigned long offset) {
        // Always apply offset. UB if base is ``nullptr`` and ``offset`` is not
        // zero, or if ``base`` is non-``nullptr`` and ``offset`` is
        // ``-reinterpret_cast<intptr_t>(base)``.
        return getelementpointer_inbounds(base, offset);
      }

      char *getelementpointer_safe(char *base, unsigned long offset) {
        // Cast pointer to integer, perform usual arithmetic addition,
        // and cast to pointer. This is legal.
        char *computed =
            reinterpret_cast<char *>(reinterpret_cast<intptr_t>(base) + offset);
        // If either the pointer becomes non-``nullptr``, or becomes
        // ``nullptr``, we must use ``computed`` result.
        if (((base == nullptr) && (computed != nullptr)) ||
            ((base != nullptr) && (computed == nullptr)))
          return computed;
        // Else we can use ``getelementpointer_inbounds()``.
        return getelementpointer_inbounds(base, offset);
      }

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
