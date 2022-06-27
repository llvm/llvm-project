# Flang 15.0.0 (In-Progress) Release Notes

> **warning**
>
> These are in-progress notes for the upcoming LLVM 15.0.0 release.
> Release notes for previous releases can be found on [the Download
> Page](https://releases.llvm.org/download.html).

## Introduction

This document contains the release notes for the Flang Fortran frontend,
part of the LLVM Compiler Infrastructure, release 15.0.0. Here we
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
* The bash wrapper script, `flang`, is renamed as `flang-to-external-fc`.

## New Compiler Flags
* Refined how `-f{no-}color-diagnostics` is treated to better align with Clang.
  In particular, both `-fcolor-diagnostics` and `-fno-color-diagnostics` are
  now available in `flang-new` (the diagnostics are formatted by default). In
  the frontend driver, `flang-new -fc1`, only `-fcolor-diagnostics` is
  available (by default, the diagnostics are not formatted). Note that this
  will only affect the diagnostics printed by driver (scanning, parsing and
  semantic diagnostics are not affected).

## Windows Support

## Fortran Language Changes in Flang

## Build System Changes

## New Issues Found


## Additional Information

Flang's documentation is located in the `flang/docs/` directory in the
LLVM monorepo.

If you have any questions or comments about Flang, please feel free to
contact us via the [mailing
list](https://lists.llvm.org/mailman/listinfo/flang-dev).
