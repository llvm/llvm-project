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

- clang used to run the actual compilation in a subprocess ("clang -cc1").
  Now compilations are done in-process by default. ``-fno-integrated-cc1``
  restores the former behavior. The ``-v`` and ``-###`` flags will print
  "(in-process)" when compilations are done in-process.

- Concepts support. Clang now supports C++2a Concepts under the -std=c++2a flag.

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
- -Wrange-loop-analysis got several improvements. It no longer warns about a
  copy being made when the result is bound to an rvalue reference. It no longer
  warns when an object of a small, trivially copyable type is copied. The
  warning now offers fixits. Excluding -Wrange-loop-bind-reference it is now
  part of -Wall. To reduce the number of false positives the diagnostic is
  disabled in macros and template instantiations.
- -Wmisleading-indentation has been added. This warning is similar to the GCC
  warning of the same name. It warns about statements that are indented as if
  they were part of a if/else/for/while statement but are not semantically
  part of that if/else/for/while.

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

* clang now defaults to ``.init_array`` on Linux. It used to use ``.ctors`` if
  the found gcc installation is older than 4.7.0. Add ``-fno-use-init-array`` to
  get the old behavior (``.ctors``).

* The behavior of the flag ``-flax-vector-conversions`` has been modified to
  more closely match GCC, as described below. In Clang 10 onwards, command lines
  specifying this flag do not permit implicit vector bitcasts between integer
  vectors and floating-point vectors. Such conversions are still permitted by
  default, however, and the default can be explicitly requested with the
  Clang-specific flag ``-flax-vector-conversions=all``. In a future release of
  Clang, we intend to change the default to ``-fno-lax-vector-conversions``.

* Improved support for ``octeon`` MIPS-family CPU. Added ``octeon+`` to
  the list of of CPUs accepted by the driver.

* For the WebAssembly target, the ``wasm-opt`` tool will now be run if it is
  found in the PATH, which can reduce code size.

* For the RISC-V target, floating point registers can now be used in inline
  assembly constraints.

New Compiler Flags
------------------

- The -fgnuc-version= flag now controls the value of ``__GNUC__`` and related
  macros. This flag does not enable or disable any GCC extensions implemented in
  Clang. Setting the version to zero causes Clang to leave ``__GNUC__`` and
  other GNU-namespaced macros, such as ``__GXX_WEAK__``, undefined.

- vzeroupper insertion on X86 targets can now be disabled with -mno-vzeroupper.
  You can also force vzeroupper insertion to be used on CPUs that normally
  wouldn't with -mvzeroupper.

- The -fno-concept-satisfaction-caching can be used to disable caching for
  satisfactions of Concepts. The C++2a draft standard does not currently permit
  this caching, but disabling it may incur significant compile-time costs. This
  flag is intended for experimentation purposes and may be removed at any time;
  please let us know if you encounter a situation where you need to specify this
  flag for correct program behavior.

- The `-ffixed-xX` flags now work on RISC-V. These reserve the corresponding
  general-purpose registers.

- RISC-V has added `-mcmodel=medany` and `-mcmodel=medlow` as aliases for
  `-mcmodel=small` and `-mcmodel=medium` respectively. Preprocessor definitions
  for `__riscv_cmodel_medlow` and `__riscv_cmodel_medany` have been corrected.

Deprecated Compiler Flags
-------------------------

The following options are deprecated and ignored. They will be removed in
future versions of Clang.

- -mmpx used to enable the __MPX__ preprocessor define for the Intel MPX
  instructions. There were no MPX intrinsics.
- -mno-mpx used to disable -mmpx and is the default behavior.
- -fconcepts-ts previously used to enable experimental concepts support. Use
  -std=c++2a instead to enable Concepts support.

- ...

Modified Compiler Flags
-----------------------

- RISC-V now sets the architecture (riscv32/riscv64) based on the value provided
  to the ``-march`` flag, overriding the target provided by ``-triple``.

- ``-flax-vector-conversions`` has been split into three different levels of
  laxness, and has been updated to match the GCC semantics:

  - ``-flax-vector-conversions=all``: This is Clang's current default, and
    permits implicit vector conversions (performed as bitcasts) between any
    two vector types of the same overall bit-width.
    Former synonym: ``-flax-vector-conversions`` (Clang <= 9).

  - ``-flax-vector-conversions=integer``: This permits implicit vector
    conversions (performed as bitcasts) between any two integer vector types of
    the same overall bit-width.
    Synonym: ``-flax-vector-conversions`` (Clang >= 10).

  - ``-flax-vector-conversions=none``: Do not perform any implicit bitcasts
    between vector types.
    Synonym: ``-fno-lax-vector-conversions``.

- ``-debug-info-kind`` now has an option ``-debug-info-kind=constructor``,
  which is one level below ``-debug-info-kind=limited``. This option causes
  debug info for classes to be emitted only when a constructor is emitted.

- RISC-V now chooses a slightly different sysroot path and defaults to using
  compiler-rt if no GCC installation is detected.

- RISC-V now supports multilibs in baremetal environments. This support does not
  extend to supporting multilib aliases.

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- Support was added for function ``__attribute__((target("branch-protection=...")))``

Windows Support
---------------

- Previous Clang versions contained a work-around to avoid an issue with the
  standard library headers in Visual Studio 2019 versions prior to 16.3. This
  work-around has now been removed, and users of Visual Studio 2019 are
  encouraged to upgrade to 16.3 or later, otherwise they may see link errors as
  below:

  .. code-block:: console

    error LNK2005: "bool const std::_Is_integral<int>" (??$_Is_integral@H@std@@3_NB) already defined

- The ``.exe`` output suffix is now added implicitly in MinGW mode, when
  Clang is running on Windows (matching GCC's behaviour)

- Fixed handling of TLS variables that are shared between object files
  in MinGW environments

- The ``-cfguard`` flag now emits Windows Control Flow Guard checks on indirect
  function calls. The previous behavior is still available with the
  ``-cfguard-nochecks`` flag. These checks can be disabled for specific
  functions using the new ``__declspec(guard(nocf))`` modifier.


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

- RISC-V now uses `target-abi` module metadata to encode the chosen psABI. This
  ensures that the correct lowering will be done by LLVM when LTO is enabled.

- An issue with lowering return types in the RISC-V ILP32D psABI has been fixed.

OpenMP Support in Clang
-----------------------

New features for OpenMP 5.0 were implemented. Use ``-fopenmp-version=50`` option to activate support for OpenMP 5.0.

- Added support for ``device_type`` clause in declare target directive.
- Non-static and non-ordered loops are nonmonotonic by default.
- Teams-based directives can be used as a standalone directive.
- Added support for collapsing of non-rectangular loops.
- Added support for range-based loops.
- Added support for collapsing of imperfectly nested loops.
- Added support for ``master taskloop``, ``parallel master taskloop``, ``master taskloop simd`` and ``parallel master taskloop simd`` directives.
- Added support for ``if`` clauses in simd-based directives.
- Added support for unified shared memory for NVPTX target.
- Added support for nested atomic and simd directives are allowed in sims-based directives.
- Added support for non temporal clauses in sims-based directives.
- Added basic support for conditional lastprivate variables

Other improvements:

- Added basic analysis for use of the uninitialized variables in clauses.
- Bug fixes.

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

- Various changes to reduce discrepancies in destructor calls between the
  generated ``CFG`` and the actual ``codegen``.

  In particular:

  - Respect C++17 copy elision; previously it would generate destructor calls
    for elided temporaries, including in initialization and return statements.

  - Don't generate duplicate destructor calls for statement expressions.

  - Fix initialization lists.

  - Fix comma operator.

  - Change printing of implicit destructors to print the type instead of the
    class name directly, matching the code for temporary object destructors.
    The class name was blank for lambdas.


Static Analyzer
---------------

- New checker: ``alpha.cplusplus.PlacementNew`` to detect whether the storage
  provided for default placement new is sufficiently large.

- New checker: ``fuchsia.HandleChecker`` to detect leaks related to Fuchsia
  handles.

- New checker: ``security.insecureAPI.decodeValueOfObjCType`` warns about
  potential buffer overflows when using ``[NSCoder decodeValueOfObjCType:at:]``

- ``deadcode.DeadStores`` now warns about nested dead stores.

- Condition values that are relevant to the occurance of a bug are far better
  explained in bug reports.

- Despite still being at an alpha stage, checkers implementing taint analyses
  and C++ iterator rules were improved greatly.

- Numerous smaller fixes.

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
