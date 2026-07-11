---
myst:
  enable_extensions:
    - attrs_block
    - substitution
---

% If you want to modify sections/contents permanently, you should modify both
% ReleaseNotes.md and ReleaseNotesTemplate.txt.

{#clang-release-releasenotestitle}
# Clang {{ (('(In-Progress) ' if env.app.tags.has('PreRelease') else '') ~ 'Release Notes') if env.config.project == 'Clang' else '|ReleaseNotesTitle|' }}

```{contents}
:depth: 2
:local:
```

Written by the [LLVM Team](https://llvm.org/)

````{only} PreRelease

```{warning}
These are in-progress notes for the upcoming Clang {{env.config.version}} release.
Release notes for previous releases can be found on
[the Releases Page](https://llvm.org/releases/).
```
````

## Introduction

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release {{env.config.release}}. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see [the LLVM
documentation](https://llvm.org/docs/ReleaseNotes.html). For the libc++ release notes,
see [this page](https://libcxx.llvm.org/ReleaseNotes.html). All LLVM releases
may be downloaded from the [LLVM releases web site](https://llvm.org/releases/).

For more information about Clang or LLVM, including information about the
latest release, please see the [Clang Web Site](https://clang.llvm.org) or the
[LLVM Web Site](https://llvm.org).

## Potentially Breaking Changes

### C/C++ Language Potentially Breaking Changes

- Clang now makes it ill-formed to try to `break` out of or `continue` a loop inside its own condition,
  increment, or init-statement in all C and C++ language modes. This means that code such as

  ```c++
  while (({ break; })) {}
  ```

  is now ill-formed. An outer loop can still be broken out of or continued so long as the inner loop is
  in the body of the outer loop:

  ```c++
  // Ok, this breaks out of the 'for' loop.
  for (;;) {
    while (({ break; true; })) {}
  }

  // Error: can't break out of the 'for' loop from within its own increment.
  for (;;({ while (({ break; true; })) {} })) {}
  ```

  This also resolves a divergence from GCC: in a construct such as

  ```c++
  for (;;) {
    while (({ break; true; })) {}
  }
  ```

  Clang would previously `break` out of the `while` loop, whereas GCC (since version 9) would
  `break` out of the `for` loop here. Now, Clang and GCC both break out of the `for` loop.

- Clang now parses line and digit directives, module names, and original filenames as unevaluated
  strings. This means that code containing strings with escape sequences such as

  ```c++
  # 1 "original\x12source.c"
  #pragma clang module import "\x41"
  # 50 "a\012.c"
  ```

  are now ill-formed.

- `__has_feature(modules)` is no longer true when just `-std=c++20` (or higher)
  is passed. It's only true if `-fmodules` is passed, which enables Clang's
  module map semantics. Objective-C++ of the form

  ```objc
  #if __has_feature(modules)
  @import Foundation;
  #else
  #import <Foundation/Foundation.h>
  #endif
  ```

  previously took the `@import` branch under `-std=c++20` even though no module
  maps were in use, which would always fail.

  `__cpp_modules` continues to be the standard macro to use to check if C++20
  modules are available.

### C++ Specific Potentially Breaking Changes

- Clang now more aggressively optimizes away stores to objects after they are
  dead. This behavior can be disabled with `-fno-lifetime-dse`.
- Clang now correctly rejects `export` declarations in module implementation
  partitions. (#GH107602)
- Template argument deduction now treats the `N` in `_BitInt(N)`
  as being of type `std::size_t` instead of `int`,
  matching the deduction of array sizes from `int(&)[N]`.
  This is a breaking change for code that depended on the previously deduced type. (#GH195033)
- Clang now rejects C++ declarations that combine the `auto` type specifier
  with another type specifier, such as `auto int`.
- Clang now rejects nested local classes defined in a different
  block scope than their parent class. (#GH193472)

### ABI Changes in This Version

- Fixed incorrect struct layout for `_BitInt` bitfields wider than 255 bits
  on MSVC targets. Internal bitfield tracking fields were changed from
  `unsigned char` to `uint64_t` to prevent overflow. This might be an ABI
  break for such structs compared to earlier Clang versions.
- Fixed a number of issues with the `__regcall` calling convention for passing
  structs on non-Windows x86-64 targets, including a crash when handling empty
  struct arguments. This changes how structs that contain arrays, floating point
  types, or `_Complex float` types are passed, and may introduce
  incompatibilities with code compiled by earlier versions of Clang that uses
  the `__regcall` calling convention on these targets. (#GH62999) (#GH98635)
- Fixed Itanium mangling for lambdas in instantiated non-static data member
  initializers by preserving the field-name closure-prefix. This changes the
  mangled names for affected lambdas. (#GH190555)
- Clang now uses MSVC-compatible manglings for supported AArch64 SVE builtin
  types when targeting the Microsoft ABI. This changes symbol names for
  affected declarations compared to previous Clang releases. (#GH196170)
- The resume and destroy functions of C++20 coroutines now use the platform C
  calling convention instead of LLVM's internal `fastcc`. This makes the
  coroutine ABI stable across LLVM versions and interoperable with other
  compilers. On most targets this is not a breaking change because `fastcc`
  and the platform C calling convention agree for `void(ptr)`. It is an ABI
  break on i686, MIPS O32, PowerPC64 ELFv1, and Lanai.
- `va_arg` on clang aarch64 msvc now reads types with 16-byte size and alignment
  (e.g. `__int128`) with their actual alignment (instead of an alignment of 8
  which was used before). Such c-variadic arguments are already passed as
  aligned, so previously reading the argument could read padding. Clang now
  matches how MSVC reads such c-variadic arguments.
- Fixed incorrect struct return when single large vector (256/512-bit) used on
  x86-64 targets. (#GH203760) The bug was introduced since Clang 21. (#GH120670)
- Clang now applies MSVC's MD5 shortening to over-long Microsoft C++ RTTI type
  descriptor name strings, matching the behavior already used for the RTTI
  symbol names. Previously the full name string was always emitted, so deeply
  nested template types (for example, ones containing local lambdas) could
  produce very large writable `.data` sections. Emitted RTTI name strings
  change only for types whose name exceeds the length limit. (#GH206313)
- Fixed Itanium mangling for lambdas in default member initializers of local
  classes to use `<local-name>` encoding, preventing mangling collisions between
  distinct local classes.

### AST Dumping Potentially Breaking Changes

- The JSON AST dump now includes all fields from `AvailabilityAttr`: `platform`,
  `introduced`, `deprecated`, `obsoleted`, `unavailable`, `message`,
  `strict`, `replacement`, `priority`, and `environment`. Previously, these
  fields were missing from the JSON output.
- Colons that appear at the end of a ParamCommentCommand name are not serialized
  as part of the name.
- AST pretty-printing now respects `PrintingPolicy::FullyQualifiedName` when
  printing `DeclRefExpr` names, including expression-form non-type template
  arguments in printed types. Previously, these references could be printed
  unqualified. (#GH206041)

### Clang Frontend Potentially Breaking Changes

- HIPSPV toolchain: `--offload-targets=spirv{32,64}` option is
  deprecated and will be removed when the new offload driver becomes
  default. The replacement for the option is
  `--offload-targets=spirv{32,64}-unknown-chipstar` when using the new
  offload driver (`--offload-new-driver`).
- The new driver (`--offload-new-driver`) is now default for all offloading
  compilations. This changes the ABI for relocatable device code. Currently,
  libraries will need to be recompiled, or used with
  (`--no-offload-new-driver`). This option will be removed in the next release.
- Clang no longer defines the `__cpp_impl_coroutine` feature test macro under the 32-bit x86 Microsoft ABI,
  as support for coroutines on this target is incomplete.
  When using coroutines on this target a warning is emmitted to indicate the lack of full support.
  That warning can be disabled with `-Wno-coroutines-unsupported-target`. (see #GH59382)

### Clang Python Bindings Potentially Breaking Changes

- Remove `CompletionString.Availability`. No libclang interfaces returned instances of it.

- `CompletionString.availability` now returns instances of `CompletionString.AvailabilityKindCompat`.

  Instances of `AvailabilityKindCompat` have the same `__str__` representation
  as the previous `CompletionChunk.Kind` and are equality-comparable with
  the existing `AvailabilityKind` enum. It will be replaced by `AvailabilityKind`
  in a future release. When this happens, the return type of `CompletionString.availability`
  will change to `AvailabilityKind`, so it is recommended to use `AvailabilityKind`
  to compare with the return values of `CompletionString.availability`.

- Remove `availabilityKinds`. In this release, uses of `availabilityKinds`
  need to be replaced by `CompletionString.AvailabilityKind`.

- `CompletionChunk.kind` now returns instances of `CompletionChunkKind`.

  Instances of `CompletionChunkKind` have the same `__str__` representation
  as the previous `CompletionChunk.Kind` for compatibility.
  These representations will be changed in a future release to match other enums.

- Remove `completionChunkKindMap`. In this release, uses of `completionChunkKindMap`
  need to be replaced by `CompletionChunkKind`.

- Move `SPELLING_CACHE` into `CompletionChunk` and change it to use
  `CompletionChunkKind` instances as keys, instead of the enum values.
  An alias is kept in the form of a `SPELLING_CACHE` variable, but it only supports
  `__getitem__` and `__contains__`. It will be removed in a future release.
  Please migrate to using `CompletionChunk.SPELLING_CACHE` instead.

- `SourceLocation` and `SourceRange` now use `NotImplemented` to delegate
  equality checks (`__eq__`) to the other object they are compared with when
  they are of different classes. They previously returned `False` when compared
  with objects of other classes.

- `TranslationUnit.get_tokens` now throws an error if both the `extent` and
  `locations` argument are passed. Previousy, `locations` took precedence.

- `_CXUnsavedFile` will be renamed to `UnsavedFile` for consistency.
  `UnsavedFile` is already available to use and existing uses should
  be adapted to refer to it instead. `_CXUnsavedFile` will be removed in a
  future release.

### OpenCL Potentially Breaking Changes

- Clang now diagnoses zero-length arrays as errors in OpenCL. OpenCL C 3.0
  section 6.11.d states that "Variable length arrays and structures with
  flexible (or unsized) arrays are not supported."

{#what-s-new-in-clang-release}
## What's New in Clang {{env.config.release}}?

### C++ Language Changes

- `__is_trivially_equality_comparable` no longer returns false for all enum types. (#GH132672)
- `auto` parameters are now available in all C++ language modes as an extension.

#### C++2d Feature Support

- Added compiler flags `-std=c++2d` and `-std=gnu++2d` for experimental C++2d implementation work.

- Clang now supports [P3733R1](https://wg21.link/p3733r1>) More named universal character escapes.
  The change is applied as a DR to all C++ language modes. (#GH203944)

#### C++2c Feature Support

- Clang now propagates `constinit` and `constexpr` in structured bindings with tuple-like initializers.

- Clang now has partial support for [P1306R5](https://wg21.link/P1306R5) Expansion Statements. Iterating expansion
  statements currently cannot be expanded and will result in a diagnostic, but other types of expansion statements work.

#### C++23 Feature Support

- Partially implement Itanium mangling for pack indexing. Partially substituted packs are not yet supported. (#GH112003)

#### C++20 Feature Support

- Clang now supports [P1857R3](https://wg21.link/p1857r3) Modules Dependency Discovery. (#GH54047)

#### C++17 Feature Support

#### Resolutions to C++ Defect Reports

- Implemented [CWG1780 Explicit instantiation/specialization of generic lambda
  operator()](https://cplusplus.github.io/CWG/issues/1780.html)
- Clang now allows omitting `typename` before a template name in a
  conversion operator, implementing [CWG2413](https://wg21.link/cwg2413).
- Member specializations can now be declared in class scope, according to the
  resolution of [CWG727](https://wg21.link/cwg727). This is still not sufficient
  to resolve that core issue.
- Clang now uses non-reference types for structured bindings whose initializer
  returns a prvalue. This resolves [CWG3135](https://wg21.link/cwg3135).

### C Language Changes

#### C2y Feature Support

- Implemented the type-specific C2y `<stdbit.h>` rotate functions with constexpr
  evaluation support:
  `stdc_rotate_left_{uc,us,ui,ul,ull}` and
  `stdc_rotate_right_{uc,us,ui,ul,ull}`.
- Implemented C2y `<stdbit.h>` memory reversal functions:
  `__builtin_stdc_memreverse8` / `stdc_memreverse8` (in-place byte
  reversal of a byte array) and `stdc_memreverse8u{8,16,32,64}` (byte-swap
  of an exact-width unsigned integer value, usable in constant expressions).

#### C23 Feature Support

- Clang now allows C23 `constexpr` struct member access through the dot operator in constant expressions. (#GH178349)
- Fixed a failing assertion when validating an invalid structure redefinition
  with a member which uses an incomplete enumeration type. (#GH190227)
- Clang now supports the C23 `wN` and `wfN` length modifiers. (#GH116962)
- Clang now recognizes the C23 `H`, `D`, and `DD` length modifiers in
  format strings and diagnoses their use because Clang does not yet support
  the corresponding decimal floating-point types, `_Decimal32`, `_Decimal64`, and `_Decimal128`. (#GH116962)
- Fixed a bug with deducing qualified inferred types with `auto`. `auto` can now
  be combined with `restrict` or `_Atomic` to form a properly-qualified type. (#GH207466)


### Objective-C Language Changes

- Clang now emits Objective-C number, array, and dictionary literals as
  compile-time constant data structures rather than runtime `objc_msgSend`
  calls on targets whose runtime supports constant literal classes. The
  feature can be disabled altogether with `-fno-objc-constant-literals`,
  or selectively per literal kind with `-fno-constant-nsnumber-literals`,
  `-fno-constant-nsarray-literals`, and
  `-fno-constant-nsdictionary-literals`.

### Non-comprehensive list of changes in this release

- Added support for floating point and pointer values in most `__atomic_`
  builtins.
- Added `__builtin_stdc_rotate_left` and `__builtin_stdc_rotate_right`
  for bit rotation of unsigned integers including `_BitInt` types. Rotation
  counts are normalized modulo the bit-width and support negative values.
  Usable in constant expressions. Implicit conversion is supported for
  class/struct types with conversion operators.
- Implemented the following C23 `<stdbit.h>` builtins with `_BitInt`
  support and constexpr evaluation:
  `__builtin_stdc_leading_zeros`, `__builtin_stdc_leading_ones`,
  `__builtin_stdc_trailing_zeros`, `__builtin_stdc_trailing_ones`,
  `__builtin_stdc_first_leading_zero`, `__builtin_stdc_first_leading_one`,
  `__builtin_stdc_first_trailing_zero`, `__builtin_stdc_first_trailing_one`,
  `__builtin_stdc_count_zeros`, `__builtin_stdc_count_ones`,
  `__builtin_stdc_has_single_bit`, `__builtin_stdc_bit_width`,
  `__builtin_stdc_bit_floor`, and `__builtin_stdc_bit_ceil`.
- Implemented the type-specific C23 `<stdbit.h>` functions with constexpr
  evaluation support:
  `stdc_leading_zeros_{uc,us,ui,ul,ull}`,
  `stdc_leading_ones_{uc,us,ui,ul,ull}`,
  `stdc_trailing_zeros_{uc,us,ui,ul,ull}`,
  `stdc_trailing_ones_{uc,us,ui,ul,ull}`,
  `stdc_first_leading_zero_{uc,us,ui,ul,ull}`,
  `stdc_first_leading_one_{uc,us,ui,ul,ull}`,
  `stdc_first_trailing_zero_{uc,us,ui,ul,ull}`,
  `stdc_first_trailing_one_{uc,us,ui,ul,ull}`,
  `stdc_count_zeros_{uc,us,ui,ul,ull}`,
  `stdc_count_ones_{uc,us,ui,ul,ull}`,
  `stdc_has_single_bit_{uc,us,ui,ul,ull}`,
  `stdc_bit_width_{uc,us,ui,ul,ull}`,
  `stdc_bit_floor_{uc,us,ui,ul,ull}`, and
  `stdc_bit_ceil_{uc,us,ui,ul,ull}`.
- A new generic bit-reverse builtin function `__builtin_bitreverseg` that
  extends bit-reversal support to all standard integers type, including
  `_BitInt`
- Added `__builtin_elementwise_clmul` for carry-less multiplication of
  integers including `_BitInt` types. This includes constexpr evaluation
  support.
- Added `__builtin_elementwise_pext` and `__builtin_elementwise_pdep` for
  parallel bit extract and parallel bit deposit operations on integers including
  `_BitInt` types. This includes constexpr evaluation support.
- Deprecated float types support from `__builtin_elementwise_max` and
  `__builtin_elementwise_min`.
- Added header `endian.h` which contains byte order helpers specified in POSIX
- Added #pragma loop licm(disable) for llvm.loop.licm.disable metadata
- Added a new `ExplicitInstantiationDecl` AST node to represent explicit
  template instantiations (e.g., `template void foo<int>();` or
  `extern template class S<int>;`). Previously, source location information
  for explicit instantiation statements was discarded after parsing. The new
  node preserves the full source range including the `extern` and `template`
  keywords, qualifiers, template arguments as written, and the declared type,
  enabling tools such as language servers and refactoring engines to accurately
  map source locations back to explicit instantiation sites.
- `typeid` on references and pointers of `final` types no longer emits a
  vtable lookup at runtime.
- Updated support for Unicode from 15.1 to 18.0.
- Linux and Windows toolchains now support Clang multilibs using
  `-fmultilib-flag=`.
- The SafeStack builtins `__builtin___get_unsafe_stack_ptr`,
  `__builtin___get_unsafe_stack_bottom`, `__builtin___get_unsafe_stack_top`,
  and `__builtin___get_unsafe_stack_start` are now deprecated. Use the
  corresponding functions from `<sanitizer/safestack_interface.h>` instead.

### New Compiler Flags

- New option `-fms-anonymous-structs` / `-fno-ms-anonymous-structs` added
  to enable or disable Microsoft's anonymous struct/union extension without
  enabling other `-fms-extensions` features (#GH177607).
- New option `--precompile-reduced-bmi` allows build system to generate a
  reduced BMI only for a C++20 importable module unit. Previously the users
  can only generate the reduced BMI as a by-product, e.g, an object files or
  a full BMI.
- New `-cc1` option `-fexperimental-overflow-behavior-types` added to
  enable parsing of the experimental `overflow_behavior` type attribute and
  type specifiers.
- New `-cl` option `/d2guardnochecks` added to match MSVC. When Windows
  Control Flow Guard (CFG) is enabled by other options, it will instruct Clang
  to emit the CFG metadata, but disable adding checks.
- New option `-fdiagnostics-show-inlining-chain` added to show inlining chain
  notes for `[[gnu::warning]]` and `[[gnu::error]]` diagnostics. When a
  function with these attributes is called from an inlined context, Clang can
  now show which functions were inlined to reach the call. When debug info is
  available (`-gline-directives-only` (implicitly enabled at `-g1`) or
  higher), accurate source locations are used; otherwise, a heuristic fallback
  is used with a note suggesting how to enable debug info for better accuracy.
- New option `-fwin-cfg-mechanism=` added to control the mechanism used by
  Control Flow Guard on Windows. Accepted values are `automatic` (default),
  `dispatch`, and `check`. The `dispatch` mechanism uses the dispatch
  function to perform indirect call checks and can improve performance, while
  `check` uses the traditional check mechanism.
- New `-cl` option `/d2guardcfgdispatch` added to match MSVC. This acts as a
  shorthand for `-fwin-cfg-mechanism=dispatch`.
- New `-cl` option `/d2guardcfgdispatch-` added to match MSVC. This acts as a
  shorthand for `-fwin-cfg-mechanism=check`.
- New option `-f[no-]strict-bool` added to control whether Clang can assume
  that `bool` values loaded from memory cannot have a bit pattern other
  than 0 or 1.
- New option `-fcrash-diagnostics-tar` added to create an archive of crash
  reproducer files for easier bug filing.
- There are a new pair of flags for riscv32 called `-mzilsd-word-align` and
  `-mzilsd-strict-align` which control whether Zilsd accesses are allowed to
  be aligned to 4-byte alignment rather than fully unaligned or fully (8-byte)
  aligned.
- New `-cl` option `/pathmap:` added to match MSVC. This option acts as a
  clang's `-ffile-prefix-map=value` and has known differences in behaviour
  with the CL's option that do not affect the functionality: nomalizes the
  macro prefix map pathes -- removes `./` and uses the target's platform-
  specific path separator character when expanding the preprocessor macros --
  `-ffile-reproducible` (but not the debug and coverage prefix maps);
  does not require `/experimental:deterministic` as by MSVC. It needed for
  removing a hostname from a mangling hash gen, but clang-cl does not use
  a hostname when generates the hashes. Known issues -- does not remap the
  source file pathes within PCH/PCM files.
- New `-cl` option `/experimental:deterministic` added to match CL's option.
  This enables warning emission on usage of non-deterministic macros `__DATE__`,
  `__TIME__` and `__TIMESTAMP__` and provides reproducable COFF's timestamp for
  the output object files.
- New `-cl` option `/d1nodatetime` added to match CL's option. This option
  undefines the standard macros `__DATE__`, `__TIME__` and `__TIMESTAMP__` to allow
  reproducable builds. These macros can be redefined from the command line if
  necessary. `/d1nodatetime-` can be used to turn this feature off if
  necessary to override the common build settings.

### Deprecated Compiler Flags

### Modified Compiler Flags

- The `-mno-outline` and `-moutline` compiler flags are now allowed on RISC-V and X86, which both support the machine outliner.
- The `-mno-outline` flag will now add the `nooutline` IR attribute, so that
  `-mno-outline` and `-moutline` objects can be mixed correctly during LTO.
- The `-fzero-call-used-regs` compiler flag is now allowed on RISC-V, only the
  "skip", "used-gpr", "used-gpr-arg", "all-gpr" and "all-gpr-arg" options are
  supported for the moment.
- Slightly changed hash id generation to get the unique linkage symbols names
  by `-unique-internal-linkage-names` option. Now it uses a path that
  normalized in favor of the target system (same as the preprocessor does
  for the file macros) and allows the reproducable IDs on any build system.
- `-fprofile-update=atomic` will now promote counter updates out of loops,
  similar to the non-atomic case ([#202487](https://github.com/llvm/llvm-project/pull/202487)).
- The `-cl` `/Brepro` option was modified to match the original CL's option
  and now defines the standard macros `__DATE__`, `__TIME__` and `__TIMESTAMP__` to
  "1". The previous functionality remains unchanged.
- The `-fms-kernel` flag will now implicitly add `-fno-delete-null-pointer-checks`.
  Still `-fdelete-null-pointer-checks` can be used to override this behavior.

### Removed Compiler Flags

### Attribute Changes in Clang

- Added new attribute `stack_protector_ignore` to opt specific local variables out of
  the analysis which determines if a function should get a stack protector. A function
  will still generate a stack protector if other local variables or command line flags
  require it.

- Added a new attribute, `[[clang::no_outline]]` to suppress outlining from
  annotated functions. This uses the LLVM `nooutline` attribute.

- Introduced a new type attribute `__attribute__((overflow_behavior))` which
  currently accepts either `wrap` or `trap` as an argument, enabling
  type-level control over overflow behavior. There is also an accompanying type
  specifier for each behavior kind via `__ob_wrap` and `__ob_trap`.

- Introduced a new function attribute `__attribute__((__personality__(...)))`
  to explicitly specify the personality routine for exception handling. THis is
  meant to be a low level tool for language runtime authors to associate a
  foreign language personality with a given function. Note that this does not
  perform any ABI validation for the personality routine.

- {doc}`ThreadSafetyAnalysis` attributes now correctly handle implicit member
  accesses in C, and parameter attributes in C++. This improves diagnostic
  precision and fixes false positives.

- The {doc}`ThreadSafetyAnalysis` attributes `guarded_by` and
  `pt_guarded_by` now accept multiple capability arguments with refined
  access semantics: *writing* requires all listed capabilities to be held
  exclusively, while *reading* requires at least one to be held. This is
  sound because any writer must hold all capabilities, so holding any one
  prevents concurrent writes.

- {doc}`ThreadSafetyAnalysis` attributes like `acquire_capability`,
  `release_capability`, `requires_capability`, `locks_excluded`,
  `try_acquire_capability`, and `assert_capability` can now be applied to
  function pointer variables and fields. The analysis checks calls through
  annotated function pointers the same way it checks direct function calls.
  Only plain function pointers are supported; pointers-to-member functions,
  blocks, or wrappers (e.g. `std::function`) are not yet supported.

- The `[[clang::unsafe_buffer_usage]]` attribute is now supported in API
  notes. For example:

  ```yaml
  Functions:
    - Name: myUnsafeFunction
      UnsafeBufferUsage: true
  ```

- When using `-Wunsafe-buffer-usage` without
  `-fsafe-buffer-usage-suggestions`, warnings are now emitted only
  once per source file. Pre-compiled code (such as PCH or module
  headers) is no longer repeatedly analyzed, as it is analyzed during
  its initial compilation. (Traditionally included headers are still
  analyzed within each translation unit that includes them). This
  behavior matches most of other `-W` diagnostics.

  When `-fsafe-buffer-usage-suggestions` is enabled, the behavior
  remains the same as before: pre-compiled code is deserialized and
  analyzed alongside the translation unit that uses it, because fix-it
  suggestion analysis requires full visibility of the translation
  unit.

- Added support for `[[msvc::forceinline]]` for functions and
  `[[msvc::forceinline_calls]]` for statements. Both are aliases to
  `[[clang::always_inline]]` with additional checks to ensure that they
  are only accepted in places where MSVC also does.

- The AMDGPU `amdgpu_num_sgpr` and `amdgpu_num_vgpr` attributes are now
  deprecated. Clang emits a `-Wdeprecated-declarations` warning when they
  are used. Use `amdgpu_waves_per_eu` instead to control SGPR and VGPR
  usage.

- Clang now allows GNU attributes between a member declarator and bit-field width. (#GH184954)

- The `[[clang::noescape]]` attribute now disallows deallocating memory
  through the annotated parameter. This information is currently not exposed to
  LLVM for optimization purposes, to prevent breaking existing adopters. It may
  instead be used by warnings and static analyses to provide more information
  about pointer lifetimes. It may be used to power optimizations in the future,
  however there are no concrete plans to do so at the moment.

- The attributes `[[clang::opencl_global_device]]` and `[[clang::opencl_global_host]]`
  are now deprecated. Clang emits a `-Wdeprecated-attributes` warning when
  they are used.

- The `modular_format` attribute now supports the `fixed` aspect for C
  ISO 18037 fixed-point `printf` specifiers.

- The `lifetime_capture_by` attribute now accepts three new spelling forms:
  `lifetime_capture_by_this`, `lifetime_capture_by_global`, and
  `lifetime_capture_by_unknown`. These replace passing `this`, `global`, and
  `unknown` as arguments to `lifetime_capture_by`; that argument form is now
  deprecated because those names can conflict with user-defined parameters.
  They will be removed in the next release. Distinct `lifetime_capture_by`
  spellings may also be combined on the same declaration, but each spelling may
  appear at most once.

- The `const` and `pure` attributes only apply to functions; they are now
  diagnosed and ignored when applied to anything else.

### Improvements to Clang's diagnostics

- Fixed bug in `-Wdocumentation` so that it correctly handles explicit
  function template instantiations (#64087).

- Fixed concept template parameters not being recognized in `-Wdocumentation`
  when mentioned in tparam comments. (#GH64087)

- `-Wunused-but-set-variable` now diagnoses file-scope variables with
  internal linkage (`static` storage class) that are assigned but never used.
  This new coverage is added under the subgroup `-Wunused-but-set-global`,
  allowing it to be disabled independently with `-Wno-unused-but-set-global`.
  (#GH148361)

- `-Wunused-template` is now part of `-Wunused` (which is enabled by `-Wall`).
  It diagnoses unused function and variable templates with internal linkage,
  which in a header is a latent ODR hazard. It can be disabled with
  `-Wno-unused-template`. (#GH202945)

- Added `-Wlifetime-safety` to enable lifetime safety analysis,
  a CFG-based intra-procedural analysis that detects use-after-free and related
  temporal safety bugs. See the
  [RFC](https://discourse.llvm.org/t/rfc-intra-procedural-lifetime-analysis-in-clang/86291)
  for more details. By design, this warning is enabled in `-Weverything`. To disable
  the analysis, use `-Wno-lifetime-safety` or `-fno-lifetime-safety`.

- Added `-Wlifetime-safety-suggestions` to enable lifetime annotation suggestions.
  This provides suggestions for function parameters that
  should be marked `[[clang::lifetimebound]]` based on lifetime analysis. For
  example, for the following function:

  ```c++
  int* p(int *in) { return in; }
  ```

  Clang will suggest:

  ```c++
  warning: parameter in intra-TU function should be marked [[clang::lifetimebound]]
  int* p(int *in) { return in; }
         ^~~~~~~
                 [[clang::lifetimebound]]
  note: param returned here
  int* p(int *in) { return in; }
                           ^~
  ```

- Added `-Wlifetime-safety-noescape` to detect misuse of `[[clang::noescape]]`
  annotation where the parameter escapes through return. For example:

  ```c++
  int* p(int *in [[clang::noescape]]) { return in; }
  ```

  Clang will warn:

  ```c++
  warning: parameter is marked [[clang::noescape]] but escapes
  int* p(int *in [[clang::noescape]]) { return in; }
         ^~~~~~~
  note: returned here
  int* p(int *in [[clang::noescape]]) { return in; }
                                               ^~
  ```

- Added `-Wlifetime-safety-dangling-field` to detect dangling field references
  when stack memory escapes to class fields. This is part of `-Wlifetime-safety`
  and detects cases where local variables or parameters are stored in fields but
  outlive their scope. For example:

  ```c++
  struct DanglingView {
    std::string_view view;
    DanglingView(std::string s) : view(s) {}  // warning: address of stack memory escapes to a field
  };
  ```

- Improved `-Wassign-enum` performance by caching enum enumerator values. (#GH176454)

- Clang now emits `-Wpsabi` diagnostics for externally visible x86-64
  function definitions that return or take AVX or AVX-512 vector types without
  enabling the corresponding target feature.

- Fixed a false negative in `-Warray-bounds` where the warning was suppressed
  when accessing a member function on a past-the-end array element.
  (#GH179128)

- Added a missing space to the FixIt for the `implicit-int` group of diagnostics and
  made sure that only one such diagnostic and FixIt is emitted per declaration group. (#GH179354)

- Fixed the Fix-It insertion point for `expected ';' after alias declaration`
  when parsing alias declarations involving a token-split `>>` sequence
  (for example, `using A = X<int>>;`). (#GH184425)

- Fixed incorrect `implicitly deleted` diagnostic for explicitly deleted
  candidate function. (#GH185693)

- The `-Wloop-analysis` warning has been extended to catch more cases of
  variable modification inside lambda expressions (#GH132038).

- Clang now emits `-Wsizeof-pointer-memaccess` when snprintf/vsnprintf use the sizeof
  the destination buffer(dynamically allocated) in the len parameter(#GH162366)

- Added `-Wmodule-map-path-outside-directory` (off by default) to warn on
  header and umbrella directory paths that use `..` to refer outside the module
  directory in module maps found via implicit search
  (`-fimplicit-module-maps`). This does not affect module maps specified
  explicitly via `-fmodule-map-file=`.

- Honour `[[maybe_unused]]` attribute on private fields.
  `-Wunused-private-field` no longer emits a warning for annotated private
  fields.

- Improved `-Wgnu-zero-variadic-macro-arguments` to suggest using
  `__VA_OPT__` if the current language version supports it(#GH188624)

- Clang now emits an error when implicitly casting a complex type to a built-in vector type. (#GH186805)

- Added `-Wnonportable-include-path-separator` (off by default) to catch
  #include directives that use backslashes as a path separator. The warning
  includes a FixIt to change all the backslashes to forward slashes, so that the
  code can automatically be made portable to other host platforms that don't
  support backslashes.

- Clang now explains why template deduction fails for explicit template arguments.

- No longer emitting a `-Wpre-c2y-compat` or extension diagnostic about use
  of octal literals with a `0o` prefix, and no longer emitting a
  `-Wdeprecated-octal-literals` diagnostic for use of octal literals without
  a `0o` prefix, when the literal is expanded from a macro defined in a
  system header. (#GH192389)

- Improved error recovery for missing semicolons after class members. Clang now avoids
  skipping subsequent valid declarations when their previous decl is missing semicolon.

- Removed the body of lambdas from some diagnostic messages.

- Fixed false positive host-device mismatch errors in discarded `if constexpr` branches for CUDA/HIP;
  such calls are now correctly skipped.

- Clang now errors when a function declaration aliases a variable or vice versa. (#GH195550)

- Added `-Wattribute-alias` to diagnose type mismatches between an alias and its aliased function. (#GH195550)

- The diagnostics around `__block` now explain why a variable cannot be marked `__block`. (#GH197213)

- Extended `-Wnonportable-include-path` to warn about trailing whitespace and dots in `#include` paths. (#GH190610)

- Clang now emits error when attribute is missing closing `]]` followed by `;;`. (#GH187223)

- Clang now rejects inline asm constraints and clobbers that contain an
  embedded null character, instead of silently truncating them. (#GH173900)

- Diagnostics for the C++11 range-based for statement now report the correct
  iterator type in notes for invalid iterator types.

- `-Wfortify-source` now warns when the constant-evaluated argument to
  `umask` has bits set outside `0777`. Those bits are silently discarded
  by the kernel, so setting them is almost always a typo (matching the
  bionic libc `diagnose_if` check).

- Improved how Unicode characters are displayed in diagnostic messages.

- `-Wtautological-pointer-compare` and `-Wpointer-bool-conversion` now
  diagnose a reference to a function (e.g. of type `void (&)()`) compared
  against or converted to a null pointer, the same as a bare function name.
  (#GH46362)

- Added a clearer diagnostic for uninitialized decomposition declarations. (#GH90107)

### Improvements to Clang's time-trace

### Improvements to Coverage Mapping

- [MC/DC] Nested expressions are handled as individual MC/DC expressions.
- "Single byte coverage" now supports branch coverage and can be used
  together with `-fcoverage-mcdc`.
- Consteval member functions are no longer emitted in coverage mappings,
  matching the existing behavior for free consteval functions. (#GH164448)

### Bug Fixes in This Version

- Fixed an assertion when comparing a fixed point type with a `_BitInt` type. (#GH196948)
- Fixed atomic boolean compound assignment; the conversion back to atomic bool would be miscompiled. (#GH33210)
- Correctly handle default template argument when establishing subsumption. (#GH188640)
- Fixed a failed assertion in the preprocessor when `__has_embed` parameters are missing parentheses. (#GH175088)
- Fix lifetime extension of temporaries in for-range-initializers in templates. (#GH165182)
- Fixed a preprocessor crash in `__has_cpp_attribute` on incomplete scoped attributes. (#GH178098)
- Fixes an assertion failure when evaluating `__underlying_type` on enum redeclarations. (#GH177943)
- Fixed an assertion failure caused by nested macro expansion during header-name lexing (`__has_embed(__has_include)`). (#GH178635)
- Clang now outputs relative paths of embeds for dependency output. (#GH161950)
- Fix the result type of a binary operation where both operands are 'void' l-values. (#GH111300)
- Fixed an assertion failure when evaluating `_Countof` on invalid `void`-typed operands. (#GH180893)
- Fixed an assertion failure in the serialized diagnostic printer when it is destroyed without calling `finish()`. (#GH140433)
- Fixed an assertion failure caused by error recovery while extending a nested name specifier with results from ordinary lookup. (#GH181470)
- Fixed a crash when parsing `#pragma clang attribute` arguments for attributes that forbid arguments. (#GH182122)
- Fixed a bug in how Clang re-transforms expressions produced from substititions
  from type aliases and concept specializations. (#GH191738) (#GH196375)
- Fixed a bug with multiple-include optimization (MIOpt) state not being preserved in some cases during lexing, which could suppress header-guard mismatch diagnostics and interfere with include-guard optimization. (#GH180155)
- Fixed a crash when normalizing constraints involving concept template parameters whose index coincided with non-concept template parameters in the same parameter mapping.
- Fixed a crash caused by accessing dependent diagnostics of a non-dependent context.
- Fixed a crash when substituting into a non-type template parameter that has a type containing an undeduced placeholder type.
- Fixed several crashes and improved diagnostics when a multidimensional subscript operator is applied to a built-in type. (#GH187800)
- Correctly diagnosing and no longer crashing when `export module foo`
  (without a semicolon) are the final tokens in a module file. (#GH187771)
- Fixed a crash in duplicate attribute checking caused by comparing constant arguments with different integer signedness. (#GH188259)
- Clang now emits an error when returning an initializer list from a lambda
  with an explicit return type of void. The diagnostic now correctly refers
  to "lambda" instead of "block". (#GH188661)
- Fixed a crash on `_BitInt(N)` arrays where 129 ≤ N ≤ 192 due to incorrect array filler lowering. (#GH189643)
- Fixed the behavior in C23 of `auto`, by emitting an error when an array type is specified for a `char *`. (#GH162694)
- Fixed an issue where an assert was thrown instead of an error if no vulkan env was specified with `--triple spirv`. (#GH189964)
- Fixed incorrect rejection of `auto` with reordered declaration specifiers in C23. (#GH164121)
- Fixed a bug where Clang fails to find instantiation of Decls in constraint checking. (#GH173086)
- Fixed a crash when assigning to an element of an `ext_vector_type` with `bool` element type. (#GH189260)
- Fixed a crash caused by declaring multiple `ownership_returns` attributes with mismatched or missing arguments. (#GH188733)
- Clang now emits an error for friend declarations of lambda members. (#GH26540)
- Fixed a crash caused by lambda capture handling in delayed default arguments. (#GH176534)
- Fixed a crash when parsing invalid `static_assert` declarations with string-literal messages (#GH187690).
- Fixed a potential stack-use-after-return issue in Clang when copy-initializing
  an array via an element-at-a-time copy loop (#GH192026)
- Fixed an issue where certain designated initializers would be rejected for constexpr variables. (#GH193373)
- Fixed `clang::Preprocessor::recomputeCurLexerKind` to avoid default fallback to `CurLexerCallback = CLK_CachingLexer;`. This prevents code-completion
  EOF handling from accidentally restoring CLK_CachingLexer while a tentative parse is still active, which could trigger a caching lexer re-entry assertion
  in clangd signature help. (#GH200677)
- Fixed incorrect expansion source ranges for synthesized tokens produced by
  feature-like builtin macros such as `__has_builtin`, which could trigger
  assertions in syntax token collection. (#GH196067)
- Fixed a crash when `#embed` is used with C++ modules (#GH195350)
- Fixed an assertion in constant evaluation when using a defaulted comparison operator in a `union`. (#GH147127)
- Fixed a bug where `-x cuda` caused clang to immediately resolve templates that should not be. (#GH200545)
- Fixed an issue where `__typeof_unqual` and `__typeof_unqual__` were rejected as a declaration specifier in block scope in C++.
- Fixed crash when checking for overflow for unary operator that can't overflow (#GH170072)
- Clang no longer handles a `" q-char-sequence "` header name as a string literal (#GH132643).
- Under `-fdollars-in-identifiers`, the `$` can now appear in user-defined-literals. (#GH173985)
- Fixed an assertion where we improperly handled implicit conversions to integral types from an atomic-type with a conversion function. (#GH201770)
- Fixed assertion failures involving code completion with delayed default arguments and exception specifications. (#GH200879)
- Fixed a regression where calling a function that takes a class-type parameter by value inside `decltype` of a concept could be incorrectly rejected when used as a non-type template argument. (#GH175831)
- Clang no longer crashes if `__attribute__((cleanup))` is applied to an invalid declaration. Furthermore, in declarations that
  contain multiple `cleanup` attributes, Clang now only uses the last one, matching GCC's behaviour (previously, Clang
  would only use the first one). A new warning that diagnoses such declarations has been added to `-Wignored-attributes`.
  (#GH191829)
- Fixed a crash in the constant evaluator when an ill-formed array new-expression whose bound could not be determined (e.g. `new int[]()`) was used in a constant expression. (#GH200139)
- Clang now defines the GCC-compatible predefined macros `__WCHAR_MIN__`, `__WINT_MIN__`, and `__SIG_ATOMIC_MIN__`. (#GH199678)
- Fix a crash in addUnsizedArray due assert not verifying we have a Base before doing checks on it. (#GH44212)

#### Bug Fixes to Compiler Builtins

- Fix a crash when passing an unresolved overload set to `__builtin_classify_type`. (#GH175589)
- Fixed a crash when calling `__builtin_allow_sanitize_check` with no arguments. (#GH183927)
- `__annotation` is now diagnosed as unsupported on non-Windows/UEFI targets, fixing a
  crash when using it with `-fms-extensions` on other platforms. (#GH184318)
- Fixed a compiler crash due to an unresolved overloaded function type when
  calling `__builtin_bit_cast`. (#GH200112)
- Clang now SFINAE friendly when the ``__reference_meows_from_temporary`` builtins
  should SFINAE friendly when the 1st type is not a reference type. (#GH206524)
- Fixed `__builtin_offsetof` incorrectly sign-extending unsigned array indices
  with the high bit set (e.g. `uint8_t` values >= 128), which produced wrong
  offset values in constant expressions. (#GH199319)


#### Bug Fixes to Attribute Support

- Fixed a behavioral discrepancy between deleted functions and private members when checking the `enable_if` attribute. (#GH175895)
- Fixed `init_priority` attribute by delaying type checks until after the type is deduced.
- Fixed a crash when a `section` attribute or `#pragma clang section` caused a
  section type conflict with a declaration whose name is not a simple identifier,
  such as a lambda's call operator. (#GH192264)
- Fixed a regression where attributed types (such as those carrying `_Nonnull`/`_Nullable` attributes)
  were not deduplicated, because the attributes' arguments were not taken into
  account when uniquing them. The duplications could substantially increase the
  size of precompiled headers and modules (PCH/PCM), and the time spent loading
  them. (#GH200961)

#### Bug Fixes to C++ Support

- Fixed a preprocessor assertion failure triggered when parsing an invalid template-id starting with `::template operator`. (#GH186582)
- Fixed a crash when a function template is defined as a non-template friend with a global scope qualifier. (#GH185341)
- Fixed a bug of incorrect template depth for abbreviated templates. (#GH200682)
- Fixed valid C++ code that uses an address-of-function-template expression,
  such as `decltype(&T::func<Args...>)`, in SFINAE checks when substituting the
  function template's trailing return type fails, such as when the return type
  contains a `decltype` probe that calls a deleted function.
- Clang now rejects constant template parameters with block pointer types, since these are not implemented anyway and would lead to crashes. (#GH189247)
- Fixed some concept bugs introduced in Clang 22 (#GH197597)
- Clang no longer reject call expressions whose type is a not-yet-deduced auto type. (#GH207565)
- Fixed a crash on error recovery when dealing with invalid templates. (#GH183075)
- Fixed a crash when instantiating `requires` expressions involving substitution failures in C++ concepts. (#GH176402)
- Concepts appearing in the require-clause of a member function no longer have access to non-public members of that class,
  or to a current class object. (#GH115838) (#GH194803) (#GH197067)
- We no longer caches invalid variable specializations. (#GH132592)
- Fixed an incorrect template argument deduction when matching packs of template
  template parameters when one of its parameters is also a pack. (#GH181166)
- Clang no longer errors on overloads with different ref-qualifiers and constraints. (#GH120812)
- Fixed a crash when a default argument is passed to an explicit object parameter. (#GH176639)
- Fixed an alias template CTAD crash.
- Improvements to typo correction involving template names. (#GH207498)
- Correctly diagnose uses of `co_await` / `co_yield` in the default argument of nested function declarations. (#GH98923)
- Fixed a crash when diagnosing an invalid static member function with an explicit object parameter (#GH177741)
- Clang incorrectly instantiated variable specializations outside of the immediate context. (#GH54439)
- Fixed a crash when pack expansions are used as arguments for non-pack parameters of built-in templates. (#GH180307)
- Fixed crash instantiating class member specializations.
- Fixed a crash during class template instantiation when a member variable
  template's type substitution fails (e.g. `typename T::type` with `T=int`),
  which left the `VarTemplateDecl` unregistered and caused a subsequent
  assertion failure when instantiating a partial specialization of that member.
  (#GH198890)
- Fix a problem where a substitution failure when evaluating a type requirement
  could directly make the program ill-formed.
- Typo correction now corrects the name qualifier for invalid template names.
- Fix a problem where pack index expressions where incorrectly being regarded as equivalent.
- Correctly diagnose narrowing in pack index expressions. (#GH205650)
- Fixed a bug where captured variables in non-mutable lambdas were incorrectly treated as mutable
  when used inside decltype in the return type. (#GH180460)
- Fixed a crash when evaluating uninitialized GCC vector/ext_vector_type vectors in `constexpr`. (#GH180044)
- Fixed a crash when `explicit(bool)` is used with an incomplete enumeration. (#GH183887)
- Fix crash-on-invalid caused by qualified forward declarations. (#GH202320)
- Fixed a crash on `typeid` of incomplete local types during template instantiation. (#GH63242), (#GH176397)
- Fixed spurious diagnostics produced when checking if constraints are equivalent for redeclarations,
  which could make the program mistakenly ill-formed.
- Fixed a crash when an immediate-invoked `consteval` lambda is used as an invalid initializer. (#GH185270)
- Fixed an assertion failure when using a global destructor with a target with a non-default program address space. (#GH186484)
- Fixed a crash when instantiating an invalid out-of-line static data member definition in a local class. (#GH176152)
- Inherited constructors in `dllexport` classes are now exported for ABI-compatible cases, matching
  MSVC behavior. Constructors with variadic arguments or callee-cleanup parameters are not yet supported
  and produce a warning. (#GH162640)
- Correctly diagnose invalid non-dependent calls in dependent contexts. (#GH135694)
- Fix initialization of GRO when GRO-return type mismatches, as part of CWG2563. (#GH98744)
- Fix an error using an initializer list with array new for a type that is not default-constructible. (#GH81157)
- Fixed a crash in invalid ``#module`` preprocessing directive. (#GH179220)
- We no longer consider conversion operators when copy-initializing from the same type. This was non
  conforming and could lead to recursive constraint satisfaction checking. (#GH149443)
- Fixed a crash in Itanium C++ name mangling for a lambda in a local class field initializer inside a constructor/destructor. (#GH176395)
- Fixed a crash when Expr::ClassifyImpl computes a classification like CL_LValue or CL_PRValue, then asserts that this
  agrees with the AST node's own value category. (#GH202693)
- Fixed crashes in Itanium C++ name mangling for lambdas with trailing requires-clauses involving requires-expressions. (#GH100774) (#GH123854)
- Fixed an invalid rejection and assertion failure while generating `operator=` for fields with the `__restrict` qualifier. (#GH37979)
- Fixed a use-after-free bug when parsing default arguments containing lambdas in declarations with template-id declarators. (#GH196725)
- Fixed missing destructor cleanups for lambda init-captures in default member
  initializers used during aggregate initialization. (#GH196469)
- Fixed a crash in constant evaluation using placement new on an array which was later initialized. (#GH196450)
- Fixed an issue where Clang incorrectly accepted invalid unqualified uses of local nested class names outside their declaring scope. (#GH184622)
- Fixed a crash when parsing invalid friend declaration with storage-class specifier. (#GH186569)
- Fixed a missing vtable for `dynamic_cast<FinalClass *>(this)` in a function template. (#GH198511)
- Fixed an assertion failure during init-list checking of an array whose element type is an incomplete class. (#GH140685)
- Fixed a crash when using a pack indexing type (e.g. ``Ts...[0]``) imported from another module. (#GH204479)
- Fixed an ODR-merging error in modules, where class-scope `using enum` declarations were not recognized as matching across module
  boundaries.  (#GH207066)

#### Bug Fixes to AST Handling

- Fixed a bug where explicit nullability property attributes were not stored in AST nodes in Objective-C. (#GH179703)
- Fixed a bug where alias CTAD, or an invalid template template parameter, could create a template with an empty template
  parameter list. This also adds asserts to prevent this from happening again.
- Fixed a crash when parsing Doxygen `@param` commands attached to invalid declarations or non-function entities. (#GH182737)
- Fixed the SourceLocation and SourceRange of reversed rewritten CXXOperatorCallExpr. (#GH192467)
- Fixed a assertion when `__block` is used on global variables in C mode. (#GH183974)
- Added missing AST nodes representing the `decltype` specifiers in destructor call to AST.
- Fixed a missing ODR violation diagnostic introduced by the inline assembly string or clobber list. (#GH198616)

#### Miscellaneous Bug Fixes

- Fixed a crash whith the AST text dumper, when dumping a reference to a
  decomposition with no bindinds. (#GH198842)
- Fixed the arguments of the format attribute on `__builtin_os_log_format`. Previously, they were off by 1.

#### Miscellaneous Clang Crashes Fixed

- Fixed a crash when attempting to jump over initialization of a variable with variably modified type. (#GH175540)
- Fixed a crash when using loop hint with a value dependent argument inside a
  generic lambda. (#GH172289)
- Fixed a crash in C++ overload resolution with `_Atomic`-qualified argument types. (#GH170433)
- Fixed a crash related to missing source locations (#GH186655)
- Fixed a crash when casting a parenthesized unresolved template-id or array section. (#GH183505)
- Fixed a crash when initializing a `constexpr` pointer with a floating-point literal in C23. (#GH180313)
- Fixed a lack of diagnostic for substitution failures in base classes when using `std::void_t`-like types.
- Fixed a crash when emitting debug info for base classes with instantiation-dependent-only types (#GH193932)
- Fixed an assertion when diagnosing address-space qualified `new`/`delete` in language-defined address spaces such as OpenCL `__local`. (#GH178319)
- Fixed an assertion failure in ObjC++ ARC when binding a rvalue reference to reference with different lifetimes (#GH178524)
- Fixed a crash when subscripting a vector type with large unsigned integer values. (#GH180563)
- Fixed a crash when attempting to diagnose incompatible conversions involving function types (#GH182534)
- Fixed a crash when evaluating `__is_bitwise_cloneable` on invalid record types. (#GH183707)
- Fixed an assertion failure when casting a function pointer with a target with a non-default program address space. (#GH186210)
- Fixed a crash when `decltype(__builtin_FUNCTION())` is used as a template type argument. (#GH167433)
- Fixed an assertion failure when parsing an invalid `decltype` specifier with missing parentheses or extra semicolons. (#GH188014)
- Fixed a crash when explicitly casting a complex type to or from an atomic complex type. (#GH172208)
- Fixed a crash when explicitly casting a scalar to an atomic complex. (#GH114885)
- Fixed an assertion failure when parsing an invalid out-of-line enum definition with template parameters. (#GH187909)
- Fixed an assertion failure on invalid template template parameter during typo correction. (#GH183983)
- Fixed an assertion failure in `isAtEndOfMacroExpansion` on macro expansions crossing the boundary of two fileIDs. (#GH115007), (#GH21755)
- Fixed an assertion failure when `__builtin_dump_struct` is used with an
  immediate-escalated callable. (#GH192846)
- Fixed a crash when passing one sized implicitly casted vector to a `abs` function. (#GH204777)
- Fixed a crash when diagnosing an invalid out-of-line definition of a member class template. (#GH201490)
- Fixed a crash in the parser when a missing semicolon after a tag definition is followed by a template-id not preceded by `::`. (#GH207992)

### OpenACC Specific Changes

### OpenCL Specific Changes

- Added support for OpenCL C 3.1 language version (`-cl-std=CL3.1`).

### Target Specific Changes

#### AMDGPU Support

- Introduced a new target specific builtin `__builtin_amdgcn_processor_is`,
  a late / deferred query for the current target processor.
- Introduced a new target specific builtin `__builtin_amdgcn_is_invocable`,
  a late / deferred query for the availability of target specific builtins.
- Initial support for gfx1310
- The `amdgpu_num_sgpr` and `amdgpu_num_vgpr` function attributes are now
  deprecated. Using them produces a `-Wdeprecated-declarations` warning. Use
  `amdgpu_waves_per_eu` instead.

#### NVPTX Support

#### X86 Support

- `march=znver6` is now supported.
- Support ISA of `AVX512BMM`.
  - Support intrinsic of `_mm512_bmacor16x16x16`.
  - Support intrinsic of `_mm512_bmacxor16x16x16`.
  - Support intrinsic of `_mm512_mask_bitrev_epi8`.
  - Support intrinsic of `_mm512_maskz_bitrev_epi8`.
  - Support intrinsic of `_mm512_bitrev_epi8`.
  - Support intrinsic of `_mm256_bmacor16x16x16`.
  - Support intrinsic of `_mm256_bmacxor16x16x16`.
  - Support intrinsic of `_mm_mask_bitrev_epi8`.
  - Support intrinsic of `_mm256_mask_bitrev_epi8`.
  - Support intrinsic of `_mm_maskz_bitrev_epi8`.
  - Support intrinsic of `_mm256_maskz_bitrev_epi8`.
  - Support intrinsic of `_mm_bitrev_epi8`.
  - Support intrinsic of `_mm256_bitrev_epi8`.
- Removed support for `AMX-TF32` (`-mamx-tf32`) and `TMMULTF32PS` instruction.

#### Arm and AArch64 Support

- On Apple AArch64 targets, `__GCC_DESTRUCTIVE_SIZE` is now `128` (down from `256`)
  to match the 128-byte cache line used by Apple cores, avoiding needless
  over-alignment. This value is implementation-defined and should not be relied upon
  in an ABI-sensitive way.

- Support has been added for the following processors (-mcpu identifiers in parenthesis):

  - Arm AGI CPU (armagicpu).
  - Hisilicon hip12 core (hip12).
  - NVIDIA Rigel core (rigel).

#### Android Support

#### Windows Support

- Clang now defines the `_MSVC_TRADITIONAL` macro as `1` when emulating MSVC
  19.15 (Visual Studio 2017 version 15.8) and later. (#GH47114)

- `-fmacro-prefix-map=` (`-ffile-prefix-map=`) now affects an anonymous namespace hash generation
  for the MSVC targets and allows deterministic symbol mangling for reproducible builds.

- Added the `-fwinx64-eh-unwind=` flag to select the x64 Windows unwind info
  version (`v1`, `v2-best-effort`, `v2-required`, or `v3`). The legacy
  `-fwinx64-eh-unwindv2=` flag is deprecated; it is still accepted and mapped
  onto the new flag as follows:

  | Legacy `-fwinx64-eh-unwindv2=` | New `-fwinx64-eh-unwind=` |
  | --- | --- |
  | `disabled` | `v1` (default; no flag forwarded) |
  | `best-effort` | `v2-best-effort` |
  | `required` | `v2-required` |

  The MSVC-compatible `/d2epilogunwind` and `/d2epilogunwindrequirev2`
  options map to `v2-best-effort` and `v2-required` respectively.

- When targeting Windows x64 with EGPR (`-mapx-features=egpr`), Clang now
  automatically enables V3 unwind info (`-fwinx64-eh-unwind=v3`) if no
  explicit unwind version was specified.

- In MSVC compatibility mode, scalar and vector deleting destructors now call
  ``__global_delete`` (or ``__global_array_delete`` for the array ``delete[]``
  path) instead of directly referencing ``::operator delete``.
  This matches MSVC's behavior and fixes ``LNK2001`` linker errors in
  environments where no global ``::operator delete`` exists. When the
  translation unit contains a ``::delete`` expression, a ``__global_delete``
  forwarding body that calls ``::operator delete`` is emitted automatically.
  Otherwise, if no body is emitted, an `/ALTERNATENAME` linker directive will
  cause the linker to use the generated `__empty_global_delete` trap function
  instead.

- Clang now supports `-std:c++26preview` for compatibility with MSVC. This enables C++26 features.

#### LoongArch Support

- DWARF fission is now compatible with linker relaxations, allowing `-gsplit-dwarf` and `-mrelax`
  to be used together when building for the LoongArch platform.

#### RISC-V Support

- Tenstorrent Ascalon D8 was renamed to Ascalon X. Use `tt-ascalon-x` with `-mcpu` or `-mtune`.
- Intrinsics were added for the `Zvabd` (RISC-V Integer Vector Absolute Difference) extension.
- Intrinsics were added for the `Zvzip` (Reordering Structured Data in Vector Registers) extension.
- A new `-mtune` syntax was added to support processor-specific tuning feature string
  Currently this new syntax is gated by the `-mexperimental-mtune-syntax` flag.

#### CUDA/HIP Language Changes

- The new offloading driver is now the default for HIP. Use
  `--no-oflfoad-new-driver` to return to the old behavior.

#### CUDA Support

- Fixed a bug where host-device ambiguities in CUDA/HIP when retrieving the
  address of specializations of templated functions that have overloads for both
  host and device. (#GH199299)

#### AIX Support

- The driver default for the linker flag `-bcdtors` now defaults to `mbr`
  (instead of `all`) which only extracts static init from archive members which
  would otherwise be referenced.
  (See <https://www.ibm.com/docs/en/aix/7.2.0?topic=l-ld-command> for details).
- The driver now uses `-lcompiler_rt` instead of `-latomic`, and the compiler-rt
  archive has been renamed from `libatomic.a` to `libcompiler_rt.a` to avoid conflicts
  between the LLVM libatomic and the GNU libatomic from the AIX toolbox as they share
  the same library name.
- Added support for `#pragma comment(copyright, "token_sequence")` on AIX.
  This directive embeds a copyright or identifying string into the compiled object file.
  The string is included in the final executable and loaded into memory at program runtime.
- The driver relaxes the restrictions on the `OBJECT_MODE` environment
  variable and now silently accepts `32_64` and `any`.

#### NetBSD Support

#### WebAssembly Support

- Fixed a crash when `__funcref` is applied to a non-function pointer type.
  (#GH118233)
- WebAssembly reference types (`__externref_t` and `__funcref` function
  pointers) now lower to the opaque IR types `target("wasm.externref")` and
  `target("wasm.funcref")` instead of `ptr addrspace(10)` /
  `ptr addrspace(20)`.
- Fixed a compiler crash at `-O2` when reference-type values were passed
  through control flow that the SLP vectorizer tried to vectorize.

#### AVR Support

#### SystemZ Support

- Add support for `#pragma export` for z/OS. This is a pragma used to export functions and variables
  with external linkage from shared libraries. It provides compatibility with the IBM XL C/C++
  compiler.

### DWARF Support in Clang

### Floating Point Support in Clang

### Fixed Point Support in Clang

### AST Matchers

- Fixed `nullPointerConstant` matcher falsely matching integer literal `0`
  in non-null-pointer contexts such as array subscripts and pointer arithmetic.
- Add `functionTypeLoc` matcher for matching `FunctionTypeLoc`.
- Add missing support for `TraversalKind` in some `addMatcher()` overloads.

### clang-format

- Add `ObjCSpaceAfterMethodDeclarationPrefix` option to control space between the
  '-'/'+' and the return type in Objective-C method declarations
- Deprecate the `BinPackParameters` and `BinPackArguments` options and replace
  them with the `PackParameters` and `PackArguments` structs (respectively) to
  unify packing behavior. Add the `BreakAfter` option to the structs, allowing
  parameter and argument lists to be formatted with one parameter/argument on each
  line if they exceed the specified count.
- Add `AfterComma` value to `BreakConstructorInitializers` to allow breaking
  constructor initializers after commas, keeping the colon on the same line.
- Extend `BreakBinaryOperations` to accept a structured configuration with
  per-operator break rules and minimum chain length gating via `PerOperator`.
- Add `AllowShortRecordOnASingleLine` option and set it to `EmptyAndAttached` for LLVM style.
- Add `BreakFunctionDeclarationParameters` option to always break before function
  declaration parameters.
- Add `EnumAssignments` option to `AlignConsecutiveAssignments` for aligning
  enum assignments without affecting other assignments.
- Add `BreakBeforeReturnType` option to break before the function return
  type.

### libclang

- Visit constraints of `auto` type to properly visit concept usages (#GH166580)
- Visit switch initializer statements (<https://bugs.kde.org/show_bug.cgi?id=415537#c2>)
- Fix crash in clang_getBinaryOperatorKindSpelling and clang_getUnaryOperatorKindSpelling
- The clang_Module_getASTFile API is deprecated and now always returns nullptr
- The clang_Cursor_getCommentRange API will now return a comment range for macro definitions that have documentation comments.
- Added CXType_PredefinedSugar for `__ptrdiff_t`, `__size_t`, and
  `__signed_size_t` types, which are no longer exposed as
  CXType_Unexposed.

### Code Completion

- Fixed a crash in code completion when using a C-Style cast with a parenthesized
  operand in Objective-C++ mode. (#GH180125)
- Fixed a crash when code completion is triggered inside an ill-formed lambda's trailing requires-clause. (#GH201632)

### Static Analyzer

- The `-analyzer-constraints` option `z3` was renamed to `unsupported-z3`
  because the Z3-based (constraint) solver was known for crashing for years now.
  Didn't receive support, so it was marked unsupported.

#### Crash and bug fixes

- Fixed `security.VAList` checker producing false positives when analyzing
  C23 code where `va_start` expands to `__builtin_c23_va_start`.
- Fixed a compiler crash when combining `_Atomic` and `__auto_type`
  in C, for example `_Atomic __auto_type x = expr`. (#GH118058)

#### Improvements

- `alpha.unix.PthreadLock` now emits path notes on lock, unlock, destroy,
  and init operations.

% comment:
% This is for the Static Analyzer.
% Using the caret `^^^` underlining for subsections:
%   - Crash and bug fixes
%   - New checkers and features
%   - Improvements
%   - Moved checkers

#### Moved checkers

- The checker `unix.cstring.UninitializedRead` is now out of alpha.

(release-notes-sanitizers)=

### Sanitizers

- UndefinedBehaviorSanitizer now supports `__ubsan_default_suppressions`.
- Sanitizer Special Case Lists (`-fsanitize-ignorelist`) now support
  Version 4 of the Special Case List format, which introduces a transition
  period for leading dot-slash (`./`) canonicalization in path matching.
  Version 4 matches both canonicalized and non-canonicalized paths but emits a
  warning for deprecated matches. Version 5 drops backward compatibility and
  requires rules to match canonicalized paths (without leading `./`).
- Sanitizer Special Case Lists (`-fsanitize-ignorelist`) and warning
  suppression mappings (`--warning-suppression-mappings`) now recognize version
  4 of the Special Case List format (indicated by `#!special-case-list-v4`).
  On Windows hosts, path matching is slash-agnostic (both forward slashes (`/`)
  and backslashes (`\`) match either path separator in both patterns and paths).

### Python Binding Changes

- Add deprecation warnings to `CompletionChunk.isKind...` methods.
  These will be removed in a future release. Existing uses should be adapted
  to directly compare equality of the `CompletionChunk` kind with
  the corresponding `CompletionChunkKind` variant.

  Affected methods: `isKindOptional`, `isKindTypedText`, `isKindPlaceHolder`,
  `isKindInformative` and `isKindResultType`.

- Add a deprecation warning to `CodeCompletionResults.results`.
  This property will become an implementation detail with changed behavior in a
  future release and should not be used directly.. Existing uses of
  `CodeCompletionResults.results` should be changed to directly use
  `CodeCompletionResults`: it nows supports `__len__` and `__getitem__`,
  so it can be used the same as `CodeCompletionResults.results`.

- Added a new helper method `get_clang_version` to the class `Config` to
  read the version string of the libclang in use.

### OpenMP Support

- Added support for `transparent` clause in task and taskloop directives.
- Added support for `use_device_ptr` clause to accept an optional
  `fallback` modifier (`fb_nullify` or `fb_preserve`) with OpenMP >= 61.
- Added support for `local` clause with declare_target directive when
  OpenMP >= 60.
- Fixed the identity element used for `reduction(* : x)` over C++ class types
  (e.g. `std::complex`). The private copy is now initialized to the
  multiplicative identity instead of being value-initialized, which previously
  produced a wrong result (the product collapsed to the additive identity).

### SYCL Support

- SYCL compilations now default to `-std=c++17` when no explicit language
  standard is specified. Standards below C++17 are rejected with a diagnostic.
- Clang now assumes default target for SYCL device compilation is 64-bit SPIR-V
  and it now diagnoses if a non-supporting target is specified via command line.
  (#GH167358)
- The SYCL runtime shared library has been renamed from `libsycl.so` to
  `libLLVMSYCL.so` to align with LLVM naming conventions.
- SYCL header include paths are now added automatically for both host and
  device compilations.
- SYCL runtime library linking is now supported on Windows. When `-fsycl` is
  specified, Clang automatically adds `/MD` if no explicit CRT flag is
  present, links the appropriate debug (`LLVMSYCLd.lib`) or release
  (`LLVMSYCL.lib`) library, and rejects static CRT flags (`/MT`,
  `/MTd`) with a diagnostic. Use `-nolibsycl` to suppress automatic
  library linking.
- Fixed `-nolibsycl` being silently ignored on Linux: the SYCL runtime
  library was unconditionally added to the link line even when the flag was
  passed.
- Added the `-fsycl-device-image-split=` option to select the granularity at
  which SYCL device code is grouped into device images. Supported values are
  `kernel` (one device image per kernel), `translation_unit` (one device image
  per translation unit), and `link_unit` (one device image per linking unit).
  The bare `-fsycl-device-image-split` flag is an alias for
  `-fsycl-device-image-split=translation_unit`, which is also the default.
- Clang now is capable of diagnosing reference kernel parameters which are not
  allowed by SYCL 2020 spec.

#### Improvements

- Improved substitution performance in concept checking. (#GH172266)
- Clang now preserves the left-hand side of a binary expression (such as an
  assignment or comparison) in a `RecoveryExpr` when the right-hand side fails
  to parse. This improves IDE features like go-to-definition in `clangd`.

## Additional Information

A wide variety of additional information is available on the [Clang web
page](https://clang.llvm.org/). The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "`clang/docs/`" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us on the [Discourse forums (Clang Frontend category)](https://discourse.llvm.org/c/clang/6).
