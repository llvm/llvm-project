<!-- If you want to modify sections/contents permanently, you should modify both
ReleaseNotes.md and ReleaseNotesTemplate.txt. -->

# Flang {{version}} {{in_progress}}Release Notes

````{only} PreRelease
```{warning}
These are in-progress notes for the upcoming LLVM {{version}} release.
Release notes for previous releases can be found on [the Download
Page](https://releases.llvm.org/download.html).
```
````

## Introduction

This document contains the release notes for the Flang Fortran frontend,
part of the LLVM Compiler Infrastructure, release {{version}}. Here we
describe the status of Flang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see [the LLVM
documentation](https://llvm.org/docs/ReleaseNotes.html). All LLVM
releases may be downloaded from the [LLVM releases web
site](https://llvm.org/releases/).

Note that if you are reading this file from a Git checkout, this
document applies to the *next* release, not the current one. To see the
release notes for a specific release, please see the [releases
page](https://llvm.org/releases/).

## Major New Features

## Bug Fixes

## Non-comprehensive list of changes in this release

## New Compiler Flags
- The `-ffpe-trap=` flag is now supported. It sets the initial floating-point
  exception halting mode for the main program to a comma-separated list of
  `invalid`, `zero`, `overflow`, `underflow`, and `inexact` (plus the
  non-standard, gfortran-compatible extension `denormal`). Use `none` or an
  empty list to disable halting. The last `-ffpe-trap=` on the command line is
  effective. The Fortran standard permits the initial halting mode to be
  processor defined (Fortran 2023, 17.6). Halting control is implemented for x86
  and glibc-based (Linux) targets; on other targets a warning is emitted and the
  option is ignored. The `denormal` exception is an x86-only extension, so
  requesting it for a non-x86 target also warns and is ignored.

## Windows Support

## Fortran Language Changes in Flang

## Build System Changes

## New Issues Found

## Additional Information

Flang's documentation is located in the `flang/docs/` directory in the
LLVM monorepo.

If you have any questions or comments about Flang, please feel free to
contact us on the [Discourse
forums](https://discourse.llvm.org/c/subprojects/flang/33).
