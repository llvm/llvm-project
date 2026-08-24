---
myst:
  enable_extensions:
    - attrs_block
    - colon_fence
    - substitution
---

% If you want to modify sections/contents permanently, you should modify both
% ReleaseNotes.md and ReleaseNotesTemplate.txt.

{#extra-clang-tools-release-releasenotestitle}
# Extra Clang Tools {{env.config.release}} {{ (('(In-Progress) ' if env.app.tags.has('PreRelease') else '') ~ 'Release Notes') }}

```{contents}
:depth: 3
:local: true
```

Written by the [LLVM Team](https://llvm.org/)

::::{only} PreRelease

:::{warning}
These are in-progress notes for the upcoming Extra Clang Tools {{env.config.version}} release.
Release notes for previous releases can be found on
[the Download Page](https://releases.llvm.org/download.html).
:::
::::

## Introduction

This document contains the release notes for the Extra Clang Tools, part of the
Clang release {{env.config.release}}. Here we describe the status of the Extra Clang Tools in
some detail, including major improvements from the previous release and new
feature work. All LLVM releases may be downloaded from the [LLVM releases web
site](https://llvm.org/releases/).

For more information about Clang or LLVM, including information about
the latest release, please see the [Clang Web Site](https://clang.llvm.org) or
the [LLVM Web Site](https://llvm.org).

Note that if you are reading this file from a Git checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the [releases page](https://llvm.org/releases/).

{#what-s-new-in-extra-clang-tools-release}
## What's New in Extra Clang Tools {{env.config.release}}?

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

### Major New Features

### Potentially Breaking Changes

- The deprecated `zircon` clang-tidy module has been removed. Users of
  `zircon-temporary-objects` should migrate to {doc}`fuchsia-temporary-objects
  <clang-tidy/checks/fuchsia/temporary-objects>`.

- In 22nd release, The `clang-tidy/ClangTidyModuleRegistry.h` header was deprecated.
  All of the symbols it used to define were moved into `clang-tidy/ClangTidyModule.h`.
  The deprecated header has been removed in this release.

### Improvements to clangd

#### Inlay hints

#### Diagnostics

#### Semantic Highlighting

#### Compile flags

#### Hover

#### Code completion

#### Code actions

- clangd now applies clang-tidy fix-it post-processing before exposing fixes.

#### Signature help

#### Cross-references

#### Objective-C

#### Miscellaneous

### Improvements to clang-doc

### Improvements to clang-query

### Improvements to clang-tidy

- Improved {program}`check_clang_tidy.py` by adding support of
  `-std=cXX-or-earlier` values, mirroring the existing `-std=cXX-or-later`.
  New construct expands to the given standard and every earlier one.

#### New checks

- New {doc}`performance-expensive-value-or
  <clang-tidy/checks/performance/expensive-value-or>` check.

  Finds calls to `value_or` (and alternative spellings `valueOr`,
  `ValueOr`) on optional types where the return type is expensive to copy.

- New {doc}`portability-avoid-pragma-comment
  <clang-tidy/checks/portability/avoid-pragma-comment>` check.

  Finds uses of `#pragma comment` and, for `lib` or `linker` comments, suggests
  using the build system for improved portability.

- New {doc}`readability-redundant-zero-initializer
  <clang-tidy/checks/readability/redundant-zero-initializer>` check.

  Finds explicit zero initializers of arrays that can be replaced with empty
  braces.

#### New check aliases

#### Changes in existing checks

- Fixed a crash in {doc}`bugprone-misplaced-operator-in-strlen-in-alloc
  <clang-tidy/checks/bugprone/misplaced-operator-in-strlen-in-alloc>` when
  checking an array new expression without a size expression.

- Fixed a crash in {doc}`bugprone-pointer-arithmetic-on-polymorphic-object
  <clang-tidy/checks/bugprone/pointer-arithmetic-on-polymorphic-object>` when
  the pointer points to an incomplete (forward-declared) type.

- Fixed a crash in {doc}`bugprone-std-namespace-modification
  <clang-tidy/checks/bugprone/std-namespace-modification>` when checking
  lambda closure types used as template arguments.

- Improved {doc}`cppcoreguidelines-pro-type-member-init
  <clang-tidy/checks/cppcoreguidelines/pro-type-member-init>` check by treating
  `std::array` the same as built-in arrays when `IgnoreArrays` option is enabled.
  
- Improved {doc}`cppcoreguidelines-use-enum-class
  <clang-tidy/checks/cppcoreguidelines/use-enum-class>` check by omitting unnamed enums from the `enum class` requirement, as previously the check suggested users an ill-formed fix.

- Improved {doc}`misc-const-correctness
  <clang-tidy/checks/misc/const-correctness>` check:

  - Fixed false positives when the pointee is written through a pointer that
    is incremented, decremented or adjusted with `+=` or `-=`, such as
    `*p++ = 0`.

  - Fixed false positives when the pointee is written through a pointer
    assignment, such as `*(p = q) = 0`.

- Improved {doc}`misc-redundant-expression
  <clang-tidy/checks/misc/redundant-expression>` by fixing false positives in
  nested expressions involving different macros or a mix of macro and
  non-macro operands.

- Fixed a crash in {doc}`modernize-raw-string-literal
  <clang-tidy/checks/modernize/raw-string-literal>` on synthetic string
  literals created for raw user-defined literal operators, such as `12_w`.

- Improved {doc}`modernize-return-braced-init-list
  <clang-tidy/checks/modernize/return-braced-init-list>` check to no longer
  rewrite the return value when the constructed type has a
  `std::initializer_list` constructor, as the braced form could select a
  different constructor.

- Fixed a crash in {doc}`modernize-use-noexcept
  <clang-tidy/checks/modernize/use-noexcept>` when analyzing malformed template
  code with an unparsed exception specification.

- Improved {doc}`performance-inefficient-algorithm
  <clang-tidy/checks/performance/inefficient-algorithm>` check to no longer
  produce a fix with the container or the searched-for value missing, such as
  `.find(43)` or `s.find()`, when either comes from a macro. The value is
  copied as written rather than with its parentheses stripped, and no fix is
  offered when an argument covers only part of a macro expansion, as it then
  has no source text of its own.

- Improved {doc}`readability-enum-initial-value
  <clang-tidy/checks/readability/enum-initial-value>` check by adding
  the {option}`AllowReferencedInitialValues` to support the
  `INT09-C-EX1` exception, allowing enumerators initialized by referencing
  another enumerator in the same enum (e.g., `last = first`).

- Improved {doc}`readability-identifier-naming
  <clang-tidy/checks/readability/identifier-naming>` check:

  - Fixed a crash when checking forward-declared classes with
    {option}`DefaultHungarianPrefix` enabled.

  - Fixed {option}`DefaultHungarianPrefix` being incorrectly diagnosed as an
    invalid option.

- Improved {doc}`readability-named-parameter
  <clang-tidy/checks/readability/named-parameter>` check by ignoring
  standard tag types (e.g. `std::in_place_t`, `std::allocator_arg_t`,
  `std::nothrow_t`, iterator tags, lock tags, etc.) that are used
  exclusively for overload resolution. Added the {option}`IgnoredTypes`
  option to allow customizing the set of ignored types.

- Improved {doc}`readability-trailing-comma
  <clang-tidy/checks/readability/trailing-comma>` check by fixing false
  positives on designated initializers, where initializer lists synthesized
  for intermediate subobjects caused the trailing comma of the enclosing
  list to be incorrectly rewritten.

- Improved {doc}`readability-use-std-min-max
  <clang-tidy/checks/readability/use-std-min-max>` check by fixing spurious
  trailing semicolons and lost comments when the `if` body has no braces.

#### Removed checks

- Removed the deprecated `zircon-temporary-objects` check. Users should migrate to
  {doc}`fuchsia-temporary-objects <clang-tidy/checks/fuchsia/temporary-objects>`.

#### Miscellaneous

### Improvements to include-fixer

### Improvements to clang-include-fixer

### Improvements to modularize

### Improvements to pp-trace

### Clang-tidy Visual Studio plugin
