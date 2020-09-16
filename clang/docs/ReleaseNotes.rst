==========================
Clang 11.0.0 Release Notes
==========================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 11.0.0. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

What's New in Clang 11.0.0?
===========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.


Recovery AST
------------

clang's AST now improves support for representing broken C++ code. This improves
the quality of subsequent diagnostics after an error is encountered. It also
exposes more information to tools like clang-tidy and clangd that consume
clang’s AST, allowing them to be more accurate on broken code.

A RecoveryExpr is introduced in clang's AST, marking an expression containing
semantic errors. This preserves the source range and subexpressions of the
broken expression in the AST (rather than discarding the whole expression).

For the following invalid code:

  .. code-block:: c++

     int NoArg(); // Line 1
     int x = NoArg(42); // oops!

clang-10 produces the minimal placeholder:

  .. code-block:: c++

     // VarDecl <line:2:1, col:5> col:5 x 'int'

clang-11 produces a richer AST:

  .. code-block:: c++

     // VarDecl <line:2:1, col:16> col:5 x 'int' cinit
     // `-RecoveryExpr <col:9, col:16> '<dependent type>' contains-errors lvalue
     //    `-UnresolvedLookupExpr <col:9> '<overloaded function>' lvalue (ADL) = 'NoArg'
     //    `-IntegerLiteral <col:15> 'int' 42

Note that error-dependent types and values may now occur outside a template
context. Tools may need to adjust assumptions about dependent code.

This feature is on by default for C++ code, and can be explicitly controlled
with `-Xclang -f[no-]recovery-ast`.

Improvements to Clang's diagnostics
-----------------------------------

- -Wpointer-to-int-cast is a new warning group. This group warns about C-style
  casts of pointers to a integer type too small to hold all possible values.

- -Wuninitialized-const-reference is a new warning controlled by 
  -Wuninitialized. It warns on cases where uninitialized variables are passed
  as const reference arguments to a function.

- ``-Wimplicit-const-int-float-conversion`` (enabled by default) is a new
  option controlled by ``-Wimplicit-int-float-conversion``.  It warns on
  implicit conversion from a floating constant to an integer type.

Non-comprehensive list of changes in this release
-------------------------------------------------

- For the ARM target, C-language intrinsics are now provided for the full Arm
  v8.1-M MVE instruction set. ``<arm_mve.h>`` supports the complete API defined
  in the Arm C Language Extensions.

- For the ARM target, C-language intrinsics ``<arm_cde.h>`` for the CDE
  instruction set are now provided.

- clang adds support for a set of  extended integer types (``_ExtInt(N)``) that
  permit non-power of 2 integers, exposing the LLVM integer types. Since a major
  motivating use case for these types is to limit 'bit' usage, these types don't
  automatically promote to 'int' when operations are done between two
  ``ExtInt(N)`` types, instead math occurs at the size of the largest
  ``ExtInt(N)`` type.

- Users of UBSan, PGO, and coverage on Windows will now need to add clang's
  library resource directory to their library search path. These features all
  use runtime libraries, and Clang provides these libraries in its resource
  directory. For example, if LLVM is installed in ``C:\Program Files\LLVM``,
  then the profile runtime library will appear at
  ``C:\Program Files\LLVM\lib\clang\11.0.0\lib\windows\clang_rt.profile-x86_64.lib``.
  To ensure that the linker can find the appropriate library, users should pass
  ``/LIBPATH:C:\Program Files\LLVM\lib\clang\11.0.0\lib\windows`` to the
  linker. If the user links the program with the ``clang`` or ``clang-cl``
  drivers, the driver will pass this flag for them.

- Clang's profile files generated through ``-fprofile-instr-generate`` are using
  a fixed hashing algorithm that prevents some collision when loading
  out-of-date profile informations. Clang can still read old profile files.

- Clang adds support for the following macros that enable the
  C-intrinsics from the `Arm C language extensions for SVE
  <https://developer.arm.com/documentation/100987/>`_ (version
  ``00bet5``, see section 2.1 for the list of intrinsics associated to
  each macro):


      =================================  =================
      Preprocessor macro                 Target feature
      =================================  =================
      ``__ARM_FEATURE_SVE``              ``+sve``
      ``__ARM_FEATURE_SVE_BF16``         ``+sve+bf16``
      ``__ARM_FEATURE_SVE_MATMUL_FP32``  ``+sve+f32mm``
      ``__ARM_FEATURE_SVE_MATMUL_FP64``  ``+sve+f64mm``
      ``__ARM_FEATURE_SVE_MATMUL_INT8``  ``+sve+i8mm``
      ``__ARM_FEATURE_SVE2``             ``+sve2``
      ``__ARM_FEATURE_SVE2_AES``         ``+sve2-aes``
      ``__ARM_FEATURE_SVE2_BITPERM``     ``+sve2-bitperm``
      ``__ARM_FEATURE_SVE2_SHA3``        ``+sve2-sha3``
      ``__ARM_FEATURE_SVE2_SM4``         ``+sve2-sm4``
      =================================  =================

  The macros enable users to write C/C++ `Vector Length Agnostic
  (VLA)` loops, that can be executed on any CPU that implements the
  underlying instructions supported by the C intrinsics, independently
  of the hardware vector register size.

  For example, the ``__ARM_FEATURE_SVE`` macro is enabled when
  targeting AArch64 code generation by setting ``-march=armv8-a+sve``
  on the command line.

  .. code-block:: c
     :caption: Example of VLA addition of two arrays with SVE ACLE.

     // Compile with:
     // `clang++ -march=armv8a+sve ...` (for c++)
     // `clang -stc=c11 -march=armv8a+sve ...` (for c)
     #include <arm_sve.h>

     void VLA_add_arrays(double *x, double *y, double *out, unsigned N) {
       for (unsigned i = 0; i < N; i += svcntd()) {
         svbool_t Pg = svwhilelt_b64(i, N);
         svfloat64_t vx = svld1(Pg, &x[i]);
         svfloat64_t vy = svld1(Pg, &y[i]);
         svfloat64_t vout = svadd_x(Pg, vx, vy);
         svst1(Pg, &out[i], vout);
       }
     }

  Please note that support for lazy binding of SVE function calls is
  incomplete. When you interface user code with SVE functions that are
  provided through shared libraries, avoid using lazy binding. If you
  use lazy binding, the results could be corrupted.

- ``-O`` maps to ``-O1`` instead of ``-O2``.
  (`D79916 <https://reviews.llvm.org/D79916>`_)

- In a ``-flto={full,thin}`` link, ``-Os``, ``-Oz`` and ``-Og`` can be used
  now. ``-Os`` and ``-Oz`` map to the -O2 pipe line while ``-Og`` maps to the
  -O1 pipeline.
  (`D79919 <https://reviews.llvm.org/D79919>`_)

- ``--coverage`` (gcov) defaults to gcov [4.8,8) compatible format now.

- On x86, ``-fpic/-fPIC -fno-semantic-interposition`` assumes a global
  definition of default visibility non-interposable and allows interprocedural
  optimizations. In produced assembly ``-Lfunc$local`` local aliases are created
  for global symbols of default visibility.

New Compiler Flags
------------------

- -fstack-clash-protection will provide a protection against the stack clash
  attack for x86, s390x and ppc64 architectures through automatic probing of
  each page of allocated stack.

- -ffp-exception-behavior={ignore,maytrap,strict} allows the user to specify
  the floating-point exception behavior. The default setting is ``ignore``.

- -ffp-model={precise,strict,fast} provides the user an umbrella option to
  simplify access to the many single purpose floating point options. The default
  setting is ``precise``.

- The default module cache has moved from /tmp to a per-user cache directory.
  By default, this is ~/.cache but on some platforms or installations, this
  might be elsewhere. The -fmodules-cache-path=... flag continues to work.

- -fpch-instantiate-templates tries to instantiate templates already while
  generating a precompiled header. Such templates do not need to be
  instantiated every time the precompiled header is used, which saves compile
  time. This may result in an error during the precompiled header generation
  if the source header file is not self-contained. This option is enabled
  by default for clang-cl.

- -fpch-codegen and -fpch-debuginfo generate shared code and/or debuginfo
  for contents of a precompiled header in a separate object file. This object
  file needs to be linked in, but its contents do not need to be generated
  for other objects using the precompiled header. This should usually save
  compile time. If not using clang-cl, the separate object file needs to
  be created explicitly from the precompiled header.
  Example of use:

  .. code-block:: console

    $ clang++ -x c++-header header.h -o header.pch -fpch-codegen -fpch-debuginfo
    $ clang++ -c header.pch -o shared.o
    $ clang++ -c source.cpp -o source.o -include-pch header.pch
    $ clang++ -o binary source.o shared.o

  - Using -fpch-instantiate-templates when generating the precompiled header
    usually increases the amount of code/debuginfo that can be shared.
  - In some cases, especially when building with optimizations enabled, using
    -fpch-codegen may generate so much code in the shared object that compiling
    it may be a net loss in build time.
  - Since headers may bring in private symbols of other libraries, it may be
    sometimes necessary to discard unused symbols (such as by adding
    -Wl,--gc-sections on ELF platforms to the linking command, and possibly
    adding -fdata-sections -ffunction-sections to the command generating
    the shared object).

- ``-fsanitize-coverage-allowlist`` and ``-fsanitize-coverage-blocklist`` are added.

- -mtls-size={12,24,32,48} allows selecting the size of the TLS (thread-local
  storage) in the local exec TLS model of AArch64, which is the default TLS
  model for non-PIC objects. Each value represents 4KB, 16MB (default), 4GB,
  and 256TB (needs -mcmodel=large). This allows large/many thread local
  variables or a compact/fast code in an executable.

- -menable-experimental-extension` can be used to enable experimental or
  unratified RISC-V extensions, allowing them to be targeted by specifying the
  extension name and precise version number in the `-march` string. For these
  experimental extensions, there is no expectation of ongoing support - the
  compiler support will continue to change until the specification is
  finalised.


Modified Compiler Flags
-----------------------

- -fno-common has been enabled as the default for all targets.  Therefore, C
  code that uses tentative definitions as definitions of a variable in multiple
  translation units will trigger multiple-definition linker errors. Generally,
  this occurs when the use of the ``extern`` keyword is neglected in the
  declaration of a variable in a header file. In some cases, no specific
  translation unit provides a definition of the variable. The previous
  behavior can be restored by specifying ``-fcommon``.
- -Wasm-ignored-qualifier (ex. `asm const ("")`) has been removed and replaced
  with an error (this matches a recent change in GCC-9).
- -Wasm-file-asm-volatile (ex. `asm volatile ("")` at global scope) has been
  removed and replaced with an error (this matches GCC's behavior).
- Duplicate qualifiers on asm statements (ex. `asm volatile volatile ("")`) no
  longer produces a warning via -Wduplicate-decl-specifier, but now an error
  (this matches GCC's behavior).
- The deprecated argument ``-f[no-]sanitize-recover`` has changed to mean
  ``-f[no-]sanitize-recover=all`` instead of
  ``-f[no-]sanitize-recover=undefined,integer`` and is no longer deprecated.
- The argument to ``-f[no-]sanitize-trap=...`` is now optional and defaults to
  ``all``.
- ``-fno-char8_t`` now disables the ``char8_t`` keyword, not just the use of
  ``char8_t`` as the character type of ``u8`` literals. This restores the
  Clang 8 behavior that regressed in Clang 9 and 10.
- -print-targets has been added to print the registered targets.
- -mcpu is now supported for RISC-V, and recognises the generic-rv32,
  rocket-rv32, sifive-e31, generic-rv64, rocket-rv64, and sifive-u54 target
  CPUs.
- ``-fwhole-program-vtables`` (along with ``-flto*``) now prepares all classes for possible whole program visibility if specified during the LTO link.
  (`D71913 <https://reviews.llvm.org/D71913>`_)

New Pragmas in Clang
--------------------

- The ``clang max_tokens_here`` pragma can be used together with
  `-Wmax-tokens <DiagnosticsReference.html#wmax-tokens>`_ to emit a warning when
  the number of preprocessor tokens exceeds a limit. Such limits can be helpful
  in limiting code growth and slow compiles due to large header files.

Attribute Changes in Clang
--------------------------

- Attributes can now be specified by clang plugins. See the
  `Clang Plugins <ClangPlugins.html#defining-attributes>`_ documentation for
  details.

Windows Support
---------------

- Don't warn about `ms_struct may not produce Microsoft-compatible layouts
  for classes with base classes or virtual functions` if the option is
  enabled globally, as opposed to enabled on a specific class/struct or
  on a specific section in the source files. This avoids needing to
  couple `-mms-bitfields` with `-Wno-incompatible-ms-struct` if building
  C++ code.

- Enable `-mms-bitfields` by default for MinGW targets, matching a similar
  change in GCC 4.7.

C Language Changes in Clang
---------------------------

- The default C language standard used when `-std=` is not specified has been
  upgraded from gnu11 to gnu17.

- Clang now supports the GNU C extension `asm inline`; it won't do anything
  *yet*, but it will be parsed.

C++ Language Changes in Clang
-----------------------------

- Clang now implements a restriction on giving non-C-compatible anonymous
  structs a typedef name for linkage purposes, as described in C++ committee
  paper `P1766R1 <http://wg21.link/p1766r1>`. This paper was adopted by the
  C++ committee as a Defect Report resolution, so it is applied retroactively
  to all C++ standard versions. This affects code such as:

  .. code-block:: c++

    typedef struct {
      int f() { return 0; }
    } S;

  Previous versions of Clang rejected some constructs of this form
  (specifically, where the linkage of the type happened to be computed
  before the parser reached the typedef name); those cases are still rejected
  in Clang 11. In addition, cases that previous versions of Clang did not
  reject now produce an extension warning. This warning can be disabled with
  the warning flag ``-Wno-non-c-typedef-for-linkage``.

  Affected code should be updated to provide a tag name for the anonymous
  struct:

  .. code-block:: c++

    struct S {
      int f() { return 0; }
    };

  If the code is shared with a C compilation (for example, if the parts that
  are not C-compatible are guarded with ``#ifdef __cplusplus``), the typedef
  declaration should be retained, but a tag name should still be provided:

  .. code-block:: c++

    typedef struct S {
      int f() { return 0; }
    } S;


OpenCL Kernel Language Changes in Clang
---------------------------------------

- Added extensions from `cl_khr_subgroup_extensions` to clang and the internal
  header.

- Added rocm device libs linking for AMDGPU.

- Added diagnostic for OpenCL 2.0 blocks used in function arguments.

- Fixed MS mangling for OpenCL 2.0 pipe type specifier.

- Improved command line options for fast relaxed math.

- Improved `atomic_fetch_min/max` functions in the internal header
  (`opencl-c.h`).

- Improved size of builtin function table for `TableGen`-based internal header
  (enabled by `-fdeclare-opencl-builtins`) and added new functionality for
  OpenCL 2.0 atomics, pipes, enqueue kernel, `cl_khr_subgroups`,
  `cl_arm_integer_dot_product`.

Changes related to C++ for OpenCL
---------------------------------

- Added `addrspace_cast` operator.

- Improved address space deduction in templates.

- Improved diagnostics of address spaces in nested pointer conversions.

ABI Changes in Clang
--------------------

- For RISC-V, an ABI bug was fixed when passing complex single-precision
  floats in RV64 with the hard float ABI. The bug could only be triggered for
  function calls that exhaust the available FPRs.


OpenMP Support in Clang
-----------------------

New features for OpenMP 5.0 were implemented.

- OpenMP 5.0 is the default version supported by the compiler. User can switch
  to OpenMP 4.5 using ``-fopenmp-version=45`` option.

- Added support for declare variant directive.

- Improved support of math functions and complex types for NVPTX target.

- Added support for parallel execution of target regions for NVPTX target.

- Added support for ``scan`` directives and ``inscan`` modifier in ``reduction``
  clauses.

- Added support for ``iterator`` construct.

- Added support for ``depobj`` construct.

- Added support for ``detach`` clauses in task-based directives.

- Added support for array shaping operations.

- Added support for cancellation constructs in ``taskloop`` directives.

- Nonmonotonic modifier is allowed with all schedule kinds.

- Added support for ``task`` and ``default`` modifiers in ``reduction`` clauses.

- Added support for strides in array sections.

- Added support for ``use_device_addr`` clause.

- Added support for ``uses_allocators`` clause.

- Added support for ``defaultmap`` clause.

- Added basic support for ``hint`` clause in ``atomic`` directives.

- Added basic support for ``affinity`` clause.

- Added basic support for ``ancestor`` modifier in ``device`` clause.

- Added support for ``default(firstprivate)`` clause. This clause is the part of
  upcoming OpenMP 5.1 and can be enabled using ``-fopenmp-version=51`` option.

- Bug fixes and optimizations.


Internal API Changes
--------------------

These are major API changes that have happened since the 10.0.0 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

- ``RecursiveASTVisitor`` no longer calls separate methods to visit specific
  operator kinds. Previously, ``RecursiveASTVisitor`` treated unary, binary,
  and compound assignment operators as if they were subclasses of the
  corresponding AST node. For example, the binary operator plus was treated as
  if it was a ``BinAdd`` subclass of the ``BinaryOperator`` class: during AST
  traversal of a ``BinaryOperator`` AST node that had a ``BO_Add`` opcode,
  ``RecursiveASTVisitor`` was calling the ``TraverseBinAdd`` method instead of
  ``TraverseBinaryOperator``. This feature was contributing a non-trivial
  amount of complexity to the implementation of ``RecursiveASTVisitor``, it was
  used only in a minor way in Clang, was not tested, and as a result it was
  buggy. Furthermore, this feature was creating a non-uniformity in the API.
  Since this feature was not documented, it was quite difficult to figure out
  how to use ``RecursiveASTVisitor`` to visit operators.

  To update your code to the new uniform API, move the code from separate
  visitation methods into methods that correspond to the actual AST node and
  perform case analysis based on the operator opcode as needed:

  * ``TraverseUnary*() => TraverseUnaryOperator()``
  * ``WalkUpFromUnary*() => WalkUpFromUnaryOperator()``
  * ``VisitUnary*() => VisiUnaryOperator()``
  * ``TraverseBin*() => TraverseBinaryOperator()``
  * ``WalkUpFromBin*() => WalkUpFromBinaryOperator()``
  * ``VisitBin*() => VisiBinaryOperator()``
  * ``TraverseBin*Assign() => TraverseCompoundAssignOperator()``
  * ``WalkUpFromBin*Assign() => WalkUpFromCompoundAssignOperator()``
  * ``VisitBin*Assign() => VisiCompoundAssignOperator()``

Build System Changes
--------------------

These are major changes to the build system that have happened since the 10.0.0
release of Clang. Users of the build system should adjust accordingly.

- clang-tidy and clang-include-fixer are no longer compiled into libclang by
  default. You can set ``LIBCLANG_INCLUDE_CLANG_TOOLS_EXTRA=ON`` to undo that,
  but it's expected that that setting will go away eventually. If this is
  something you need, please reach out to the mailing list to discuss possible
  ways forward.

clang-format
------------

- Option ``IndentExternBlock`` has been added to optionally apply indenting inside ``extern "C"`` and ``extern "C++"`` blocks.

- ``IndentExternBlock`` option accepts ``AfterExternBlock`` to use the old behavior, as well as Indent and NoIndent options, which map to true and false, respectively.

  .. code-block:: c++

    Indent:                       NoIndent:
     #ifdef __cplusplus          #ifdef __cplusplus
     extern "C" {                extern "C++" {
     #endif                      #endif

          void f(void);          void f(void);

     #ifdef __cplusplus          #ifdef __cplusplus
     }                           }
     #endif                      #endif

- Option ``IndentCaseBlocks`` has been added to support treating the block
  following a switch case label as a scope block which gets indented itself.
  It helps avoid having the closing bracket align with the switch statement's
  closing bracket (when ``IndentCaseLabels`` is ``false``).

  .. code-block:: c++

    switch (fool) {                vs.     switch (fool) {
    case 1:                                case 1: {
      {                                      bar();
         bar();                            } break;
      }                                    default: {
      break;                                 plop();
    default:                               }
      {                                    }
        plop();
      }
    }

- Option ``ObjCBreakBeforeNestedBlockParam`` has been added to optionally apply
  linebreaks for function arguments declarations before nested blocks.

- Option ``InsertTrailingCommas`` can be set to ``TCS_Wrapped`` to insert
  trailing commas in container literals (arrays and objects) that wrap across
  multiple lines. It is currently only available for JavaScript and disabled by
  default (``TCS_None``).

- Option ``BraceWrapping.BeforeLambdaBody`` has been added to manage lambda
  line break inside function parameter call in Allman style.

  .. code-block:: c++

      true:
      connect(
        []()
        {
          foo();
          bar();
        });

      false:
      connect([]() {
          foo();
          bar();
        });

- Option ``AlignConsecutiveBitFields`` has been added to align bit field
  declarations across multiple adjacent lines

  .. code-block:: c++

      true:
        bool aaa  : 1;
        bool a    : 1;
        bool bb   : 1;

      false:
        bool aaa : 1;
        bool a : 1;
        bool bb : 1;

- Option ``BraceWrapping.BeforeWhile`` has been added to allow wrapping
  before the ```while`` in a do..while loop. By default the value is (``false``)

  In previous releases ``IndentBraces`` implied ``BraceWrapping.BeforeWhile``.
  If using a Custom BraceWrapping style you may need to now set
  ``BraceWrapping.BeforeWhile`` to (``true``) to be explicit.

  .. code-block:: c++

      true:
      do {
        foo();
      }
      while(1);

      false:
      do {
        foo();
      } while(1);


.. _release-notes-clang-static-analyzer:

Static Analyzer
---------------

- Improved the analyzer's understanding of inherited C++ constructors.

- Improved the analyzer's understanding of dynamic class method dispatching in
  Objective-C.

- Greatly improved the analyzer's constraint solver by better understanding
  when constraints are imposed on multiple symbolic values that are known to be
  equal or known to be non-equal. It will now also efficiently reject impossible
  if-branches between known comparison expressions.

- Added :ref:`on-demand parsing <ctu-on-demand>` capability to Cross Translation
  Unit (CTU) analysis.

- Numerous fixes and improvements for the HTML output.

- New checker: :ref:`alpha.core.C11Lock <alpha-core-C11Lock>` and
  :ref:`alpha.fuchsia.Lock <alpha-fuchsia-lock>` checks for their respective
  locking APIs.

- New checker: :ref:`alpha.security.cert.pos.34c <alpha-security-cert-pos-34c>`
  finds calls to ``putenv`` where a pointer to an autmoatic variable is passed
  as an argument.

- New checker: :ref:`webkit.NoUncountedMemberChecker
  <webkit-NoUncountedMemberChecker>` to enforce the existence of virtual
  destructors for all uncounted types used as base classes.

- New checker: :ref:`webkit.RefCntblBaseVirtualDtor
  <webkit-RefCntblBaseVirtualDtor>` checks that only ref-counted types
  are used as class members, not raw pointers and references to uncounted
  types.

- New checker: :ref:`alpha.cplusplus.SmartPtr <alpha-cplusplus-SmartPtr>` check
  for dereference of null smart pointers.

- Moved ``PlacementNewChecker`` out of alpha, making it enabled by default.

- Added support for multi-dimensional variadic arrays in ``core.VLASize``.

- Added a check for insufficient storage in array placement new calls, as well
  as support for alignment variants to ``cplusplus.PlacementNew``.

- While still in alpha, ``alpha.unix.PthreadLock``, the iterator and container
  modeling infrastructure, ``alpha.unix.Stream``, and taint analysis were
  improved greatly. An ongoing, currently off-by-default improvement was made on
  the pre-condition modeling of several functions defined in the POSIX standard.

- Improved the warning messages of several checkers.

- Fixed a few remaining cases of checkers emitting reports under incorrect
  checker names, and employed a few restrictions to more easily identify and
  avoid such errors.

- Moved several non-reporting checkers (those that model, among other things,
  standard functions, but emit no diagnostics) to be listed under
  ``-analyzer-checker-help-developer`` instead of ``-analyzer-checker-help``.
  Manually enabling or disabling checkers found on this list is not supported
  in production.

- Numerous fixes for crashes, false positives and false negatives in
  ``unix.Malloc``, ``osx.cocoa.NSError``, and several other checkers.

- Implemented a dockerized testing system to more easily determine the
  correctness and performance impact of a change to the static analyzer itself.
  The currently beta-version tool is found in
  ``(llvm-project repository)/clang/utils/analyzer/SATest.py``.

.. _release-notes-ubsan:


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
