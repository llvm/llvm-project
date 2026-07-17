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

### C++ Specific Potentially Breaking Changes

### ABI Changes in This Version

- Except on PlayStation, Clang now derives the x86-64 System V AVX ABI level
for 256- and 512-bit vector arguments and returns from effective per-function
target features. Features and `arch=` CPUs that imply AVX or AVX512F are
honored, and calls use the caller's features, matching GCC. Per-function
features cannot lower the translation-unit ABI level;
`-fclang-abi-compat=23` restores the previous behavior. (#GH193298)

### AST Dumping Potentially Breaking Changes

### Clang Frontend Potentially Breaking Changes

### Clang Python Bindings Potentially Breaking Changes

### OpenCL Potentially Breaking Changes

## What's New in Clang {{env.config.release}}?

### C++ Language Changes

#### C++2d Feature Support

#### C++2c Feature Support

- Clang now supports [P3533R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3533r2.html) (constexpr virtual inheritance).

#### C++23 Feature Support

#### C++20 Feature Support

#### C++17 Feature Support

#### Resolutions to C++ Defect Reports

### C Language Changes

#### C2y Feature Support

- Clang now supports C2y's new syntax for `if` and `switch` statements with
  initializer and condition variables, as specified in
  [N3356](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3356.htm)_. For
  example:

```c
  if (bool x = true; x) {
    // ...
  }

  if (bool x = true) {
    // ...
  }

  // attribute list on declarations are also supported
  switch ([[maybe_unused]] int x = 1) {
  default:
    // ...
  }

  if (bool x [[maybe_unused]] = true; x) {
    // ...
  }
```

#### C23 Feature Support

### Objective-C Language Changes

### Non-comprehensive list of changes in this release

### New Compiler Flags

### Deprecated Compiler Flags

### Modified Compiler Flags

### Removed Compiler Flags

### Attribute Changes in Clang

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

- Added `-Wstringop-overread` to warn when `memcpy`, `memmove`, `memcmp`,
  and related builtins read more bytes than the source buffer size (#GH83728).

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


### Improvements to Clang's time-trace

### Improvements to Coverage Mapping

### Bug Fixes in This Version

- Fixed a constraint comparison bug in partial ordering. (#GH182671)
- Fixed a rejected-valid case that used an explicit object parameter in an out-of-line definition of a nested class member. (#GH136472)

#### Bug Fixes to Compiler Builtins

#### Bug Fixes to Attribute Support

- The `counted_by`/`counted_by_or_null` diagnostic that rejects a pointer whose
  pointee is a struct with a flexible array member (e.g.
  ``struct with_fam * __sized_by(size) ptr;``) was incorrectly also applied to
  the `sized_by`/`sized_by_or_null` attributes. Because `sized_by` and
  `sized_by_or_null` describe the size in bytes rather than a count of elements,
  they are now correctly accepted on such pointers.

#### Bug Fixes to C++ Support

-Fixed an issue where we tried to compare invalid NTTPs for variable declarations, which ended up in hitting an assertion with a constrained non-plain-auto NTTP, which we don't quite implement yet. (#GH208658)

- Fixed a crash when a using-declaration naming an unresolvable member of a
  dependent base was shadowed by an invalid using-declaration. (#GH209427)

#### Bug Fixes to AST Handling

- Fixed a non-deterministic ordering of unused local typedefs that made
  serialized PCH/AST files and `-Wunused-local-typedef` diagnostics
  non-reproducible across runs. (#GH209639)

#### Miscellaneous Bug Fixes

#### Miscellaneous Clang Crashes Fixed

### OpenACC Specific Changes

### OpenCL Specific Changes

- Extensions ``cl_khr_extended_bit_ops``, ``cl_khr_integer_dot_product``,
  ``cl_khr_subgroup_extended_types``, ``cl_khr_subgroup_rotate``,
  ``cl_khr_subgroup_shuffle``, and ``cl_khr_subgroup_shuffle_relative`` are
  promoted to core features in OpenCL C 3.1. A target claiming OpenCL C 3.1
  conformance without supporting one of these features is now diagnosed.

### Target Specific Changes

#### AMDGPU Support

- Deprecated the following builtins in favor of `__builtin_amdgcn_ballot_w32` or
  `__builtin_amdgcn_ballot_w64`:
  - `__builtin_amdgcn_uicmp`
  - `__builtin_amdgcn_uicmpl`
  - `__builtin_amdgcn_sicmpl`
  - `__builtin_amdgcn_fcmp`
  - `__builtin_amdgcn_fcmpf`

#### NVPTX Support

#### X86 Support

#### Arm and AArch64 Support

- On AArch64 Windows targets, `-mbranch-protection=standard` and `-mbranch-protection=pac-ret`
  now uses the B-key by default.

#### Android Support

#### Windows Support

#### LoongArch Support

#### RISC-V Support

#### CUDA/HIP Language Changes

#### CUDA Support

#### AIX Support

#### NetBSD Support

#### WebAssembly Support

#### AVR Support

#### SystemZ Support

### DWARF Support in Clang

### Floating Point Support in Clang

### Fixed Point Support in Clang

### AST Matchers

### clang-format

### libclang

### Code Completion

### Static Analyzer

#### Crash and bug fixes

% comment:
% This is for the Static Analyzer.
% Use `####` headings for subsections:
%   - Crash and bug fixes
%   - New checkers and features
%   - Improvements
%   - Moved checkers
%   - Diagnostic changes

#### Improvements

- The lock-order-reversal check in ``alpha.unix.PthreadLock`` is now disabled by default.
  It can be re-enabled with the ``WarnOnLockOrderReversal`` option.

#### Moved checkers

#### Diagnostic changes

- For self-assignments during initialization (`T v = v;`), `core.uninitialized.Assign` will not report them as uninitialized accesses (except C++ reference types), and the checks will be delayed until the first accesses of these variables; `deadcode.DeadStores` will not report them as dead stores. (#GH187530)

(release-notes-sanitizers)=

### Sanitizers

### Python Binding Changes

### OpenMP Support

### SYCL Support

#### Improvements

## Additional Information

A wide variety of additional information is available on the [Clang web
page](https://clang.llvm.org/). The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "`clang/docs/`" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us on the [Discourse forums (Clang Frontend category)](https://discourse.llvm.org/c/clang/6).
