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

#### Bug Fixes to AST Handling

#### Miscellaneous Bug Fixes

#### Miscellaneous Clang Crashes Fixed

### OpenACC Specific Changes

### OpenCL Specific Changes

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
