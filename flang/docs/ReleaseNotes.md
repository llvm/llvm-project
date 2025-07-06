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

* Initial support for VOLATILE variables and procedure interface arguments has been added.

## Bug Fixes

## Non-comprehensive list of changes in this release

## New Compiler Flags

* -floop-interchange is now recognized by flang.
* -floop-interchange is enabled by default at -O2 and above.
* -fveclib=libmvec is supported for AArch64 (same as Flang/x86 and
  Clang/AArch64) (requires GLIBC 2.40 or newer)

## Windows Support

## Fortran Language Changes in Flang

## Build System Changes

 * The FortranRuntime library has been renamed to `flang_rt.runtime`.

 * The FortranFloat128Math library has been renamed to `flang_rt.quadmath`.

 * The CufRuntime_cuda_${version} library has been renamed to
   `flang_rt.cuda_${version}`.

 * The Fortran Runtime library has been move to a new top-level directory
   named "flang-rt". It now supports the LLVM_ENABLE_RUNTIMES mechanism to
   build Flang-RT for multiple target triples. libflang_rt.runtime.{a|so} will
   now be emitted into Clang's per-target resource directory
   (next to libclang_rt.*.*) where it is also found by Flang's driver.

  * Flang on AArch64 now always depends on compiler-rt to provide the
    `__trampoline_setup` function. This dependency will be automatically added
    in in-tree builds with the AArch64 target, but compiler-rt will need to be
    manually added to LLVM builds when building flang out-of-tree.

## New Issues Found


## Additional Information

Flang's documentation is located in the `flang/docs/` directory in the
LLVM monorepo.

If you have any questions or comments about Flang, please feel free to
contact us on the [Discourse
forums](https://discourse.llvm.org/c/subprojects/flang/33).
