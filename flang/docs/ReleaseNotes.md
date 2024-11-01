# Flang |version| (In-Progress) Release Notes

> **warning**
>
> These are in-progress notes for the upcoming LLVM |version| release.
> Release notes for previous releases can be found on [the Download
> Page](https://releases.llvm.org/download.html).

## Introduction

This document contains the release notes for the Flang Fortran frontend,
part of the LLVM Compiler Infrastructure, release |version|. Here we
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

* Flang now supports loading LLVM pass plugins with the `-fpass-plugin` option
  which is also available in clang. The option mimics the behavior of the
  corresponding option in clang and has the same capabilities and limitations.
* Flang also supports statically linked LLVM pass extensions. Projects can be
  linked statically into `flang-new` if the cmake command includes
  `-DLLVM_${NAME}_LINK_INTO_TOOLS=ON`. This behavior is also similar to clang.

## Bug Fixes

## Non-comprehensive list of changes in this release
* The bash wrapper script, `flang`, is renamed as `flang-to-external-fc`.
* In contrast to Clang, Flang will not default to using `-fpie` when linking
  executables. This is only a temporary solution and the goal is to align with
  Clang in the near future. First, however, the frontend driver needs to be
  extended so that it can generate position independent code (that requires
  adding support for e.g. `-fpic` and `-mrelocation-model` in `flang-new
  -fc1`). Once that is available, support for the `-fpie` can officially be
  added and the default behaviour updated.

## New Compiler Flags
* Refined how `-f{no-}color-diagnostics` is treated to better align with Clang.
  In particular, both `-fcolor-diagnostics` and `-fno-color-diagnostics` are
  now available in `flang-new` (the diagnostics are formatted by default). In
  the frontend driver, `flang-new -fc1`, only `-fcolor-diagnostics` is
  available (by default, the diagnostics are not formatted).

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
