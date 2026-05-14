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

- The warning flags with prefixes -Wopen-mp and -Wopen-acc have been deprecated in favor of corrected spellings with the respective prefixes -Wopenmp and -Wopenacc. Removal of the deprecated options is planned for LLVM 25 (July 2027).

- The pedantic flag now takes an optional argument, a Fortran standard: f77, f90, f95, f2003, f2008, f2018, f2023, and f202Y. The behavior of the pedantic flag without an argument remains unchanged. The pedantic flag with an argument warns on all of the same things that the pedantic flag without an argument warns. Additionally, when passed a standard as an argument, the pedantic flag warns about code not conforming to that standard. Currently, the only additional warnings are for the different versions of `SYSTEM_CLOCK` in the various standards.

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
