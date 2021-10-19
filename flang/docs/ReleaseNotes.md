# Flang 12.0.0 (In-Progress) Release Notes

> **warning**
>
> These are in-progress notes for the upcoming LLVM 12.0.0 release.
> Release notes for previous releases can be found on [the Download
> Page](https://releases.llvm.org/download.html).

## Introduction

This document contains the release notes for the Flang Fortran frontend,
part of the LLVM Compiler Infrastructure, release 12.0.0. Here we
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

## Known Issues

These are issues that couldn't be fixed before the release. See the bug
reports for the latest status.

 *   ...

## Introducing Flang

Flang is LLVM's Fortran front end and is new for the LLVM 11 release.

Flang is still a work in progress for this release and is included for
experimentation and feedback.

Flang is able to parse a comprehensive subset of the Fortran language
and check it for correctness. Flang is not yet able to generate LLVM IR
for the source code and thus is unable to compile a running binary.

Flang is able to unparse the input source code into a canonical form and
emit it to allow testing. Flang can also invoke an external Fortran
compiler on this canonical input.

Flang's parser has comprehensive support for:
 * Fortran 2018
 * OpenMP 4.5
 * OpenACC 3.0

Interested users are invited to try to compile their Fortran codes with
flang in and report any issues in parsing or semantic checking in
[bugzilla](https://bugs.llvm.org/enter_bug.cgi?product=flang).

### Major missing features

 * Flang is not supported on Windows platforms.

## Using Flang

Usage: `flang hello.f90 -o hello.bin`

By default, Flang will parse the Fortran file `hello.f90` then unparse it to a
canonical Fortran source file. Flang will then invoke an external
Fortran compiler to compile this source file and link it, placing the
resulting executable in `hello.bin`.

To specify the external Fortran compiler, set the `F18_FC` environment
variable to the name of the compiler binary and ensure that it is on your
`PATH`. The default value for `F18_FC` is `gfortran`.

When invoked with no source input, Flang will wait for input on stdin.
When invoked in this way, Flang performs the same actions as if
called with `-fdebug-measure-parse-tree -funparse` and does not invoke
`F18_FC`.

For a full list of options that Flang supports, run `flang --help`.

## Additional Information

Flang's documentation is located in the `flang/docs/` directory in the
LLVM monorepo.

If you have any questions or comments about Flang, please feel free to
contact us via the [mailing
list](https://lists.llvm.org/mailman/listinfo/flang-dev).
