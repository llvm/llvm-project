<!-- If you want to modify sections/contents permanently, you should modify both
ReleaseNotes.md and ReleaseNotesTemplate.txt. -->

# Flang {{version}} {{in_progress}}Release Notes

::::{only} PreRelease
:::{warning}
These are in-progress notes for the upcoming LLVM {{version}} release.
Release notes for previous releases can be found on [the Download
Page](https://releases.llvm.org/download.html).
:::
::::

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

- Added support for the OpenMP implementation-defined extension sentinels
  (OpenMP 5.2, section 3.1): `!$omx`, `c$omx` and `*$omx` in fixed source form
  and `!$ompx` in free source form. These sentinels are recognized like their
  `omp` counterparts when OpenMP is enabled.
  
- Change source path in -Rpass remarks (e.g., -Rpass=loop-vectorize) from a
  (mostly) full path to clang's behavior which is to use the source filename
  as specified on the command line (except that ./foo.f90 removes the ./
  prefix).

- The legacy array-value operations (`fir.array_load`, `fir.array_fetch`,
  `fir.array_update`, `fir.array_modify`, `fir.array_access`,
  `fir.array_amend`, `fir.array_merge_store`) have been removed from FIR,
  together with the `array-value-copy` pass that legalized them and its
  `-mmlir -disable-avc` option. Nothing in flang has produced these
  operations since the legacy (non-HLFIR) expression lowering was deleted.
  Downstream projects that still construct them must migrate to HLFIR (or
  their own legalization) before rebasing. `fir.array_coor` is unrelated
  and remains supported.

- Added support for compressed DWARF debug sections. Flang now supports
  compressing DWARF debug info in ELF object files using zlib or zstd,
  reducing debug information size in compiled binaries.

- The FIR loop invariant code motion pass (`flang-licm`) is now enabled by
  default at optimization levels above `-O0`. It can be turned off with
  `-mmlir -disable-fir-licm`. The `-mmlir -enable-fir-licm` option that
  previously opted into the pass has been removed.

- Named constants (`PARAMETER`) now appear in the debug information, so a
  debugger can print them by name. A constant is described only in the
  compilation unit that defines it: one declared in a module is described
  where that module is compiled, and one declared in a procedure is local to
  that unit. Constants of an intrinsic module such as `iso_fortran_env` are
  not described yet, because no compilation unit defines them.

- A reference with a constant subscript that is out of range is now accepted with
  a warning instead of being rejected with an error. A subscript is required to be
  within its bounds only when the reference is executed (F'2023 9.5.3.1 paragraph
  2), and that cannot be determined in general, so programs that keep such a
  reference in a branch or procedure that never runs are no longer rejected. The
  same applies to array section endpoints, but not to cosubscripts, which remain
  errors. Use `-fno-out-of-bounds-subscripts` to get an error again, or
  `-Wno-out-of-bounds-subscripts` to silence the warning.

## New Compiler Flags
- Added `-fno-out-of-bounds-subscripts`, which restores the previous behavior of
  rejecting an out-of-range constant subscript with an error. See the entry above
  for the change in default behavior.

- Added the gfortran-compatible `-ffpe-trap=` flag, which sets the initial
  floating-point exception halting mode of the main program. It takes a
  comma-separated list of `invalid`, `zero`, `overflow`, `underflow`, `inexact`,
  and the extension `denormal`, or `none` to disable halting. See the Flang
  command line reference for the supported targets and details.

- Added `-gz` and `-gz=<format>` flags to enable compression of DWARF debug
  sections. Supported formats are `zlib`, `zstd`, and `none`.

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
