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

- Fixed `fir::getTypeSizeAndAlignment` returning the wrong allocation size for
  **packed `fir::RecordType`s** (produced by the AIX lowering of `BIND(C)`
  derived types, or declared directly in textual FIR). Fields in a packed
  record are placed back-to-back using each component's allocation size
  (`alignTo(storeSize, ABIalign)`), not its raw store size, and the record's
  ABI alignment is 1. For example, a packed `{i32, f64}` on x86-64 now
  correctly reports 12 bytes instead of 16.
  ([#220377](https://github.com/llvm/llvm-project/pull/220377))

- Fixed `fir::getTypeSizeAndAlignment` omitting **tail padding** from unpacked
  derived types. The returned size is now rounded up to the record's own ABI
  alignment, matching the allocation extent used by array element strides, CUDA
  shared-memory layout, and stack/heap allocation placement. For example,
  `{i32, i8}` (store size 5 bytes, align 4) now correctly reports 8 bytes
  instead of 5.
  ([#220377](https://github.com/llvm/llvm-project/pull/220377))

- Fixed a **`BIND(C)` / `VALUE` argument-passing ABI bug** on SystemZ:
  derived types whose allocation size fits in a GPR were incorrectly passed
  indirectly (by reference) instead of as an integer register value, because
  `getTypeSizeAndAlignment` was returning the unpadded store size rather than
  the allocation size. For example, `{i32, i8}` (allocation size 8 bytes) is
  now correctly passed as `i64`, and `{i16, i8}` (4 bytes) as `i32`, matching
  the C ABI.
  Fortran programs with `BIND(C)` `VALUE` derived-type arguments of these shapes
  that interoperate with C were already producing incorrect results; programs
  compiled entirely in Fortran that relied on the old (incorrect) convention
  must be recompiled.
  ([#220377](https://github.com/llvm/llvm-project/pull/220377))

- Fixed a **`BIND(C)` / `VALUE` argument-passing ABI bug** on PPC64le:
  derived types were classified using the unpadded store size rather than the
  allocation size, producing the wrong number of GPR slots. The argument was
  already passed by value; only the slot count was wrong. For example,
  `{f128, i8}` (allocation size 32 bytes) is now correctly passed as
  `[4 x i64]` instead of `[3 x i64]`, matching the C ABI.
  Fortran programs with `BIND(C)` `VALUE` derived-type arguments of these shapes
  that interoperate with C were already producing incorrect results; programs
  compiled entirely in Fortran that relied on the old (incorrect) convention
  must be recompiled.
  ([#220377](https://github.com/llvm/llvm-project/pull/220377))

- Fixed the `TRANSFER` intrinsic inline path to compare **stored-representation
  widths** (excluding outer tail padding) rather than allocation sizes when
  deciding whether to inline a load instead of calling the runtime. This
  corrects two path-selection errors: inlining when the store sizes differ
  (which loaded the wrong number of bytes) and falling back to the runtime
  when the store sizes match (a missed-inlining regression). The inline path
  now also byte-copies record data into result-aligned storage so that
  internal padding bytes (e.g. in `BIND(C)` records) are preserved,
  satisfying the F2023 16.9.212 requirement that the result's physical
  representation be identical to the source's when both have the same length.
  ([#220377](https://github.com/llvm/llvm-project/pull/220377))


## Non-comprehensive list of changes in this release

- Added support for the OpenMP implementation-defined extension sentinels
  (OpenMP 5.2, section 3.1): `!$omx`, `c$omx` and `*$omx` in fixed source form
  and `!$ompx` in free source form. These sentinels are recognized like their
  `omp` counterparts when OpenMP is enabled.
  
- Change source path in -Rpass remarks (e.g., -Rpass=loop-vectorize) from a
  (mostly) full path to clang's behavior which is to use the source filename
  as specified on the command line (except that ./foo.f90 removes the ./
  prefix).

- Fortran-standard-compliant reassociation within individual `REAL` and
  `COMPLEX` sum expressions is now enabled by default at all optimization
  levels. This may change exact floating-point results. Flang users can
  restore left-to-right evaluation with `-fno-fp-sum-reassociation`.

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
