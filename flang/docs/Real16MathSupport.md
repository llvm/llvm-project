<!--===- docs/Real16MathSupport.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Flang support for REAL(16) math intrinsics

To support most `REAL(16)` (i.e. 128-bit float) math intrinsics Flang relies
on an external library providing the implementation. There are two choices,
selected with the `FLANG_RUNTIME_F128_MATH_LIB` CMake option.

## `-DFLANG_RUNTIME_F128_MATH_LIB=libm`

On a glibc of version 2.26 or later, the `*f128` entry points are exported by
`libm` itself, and this option builds `libflang_rt.quadmath` against those.
No third-party library is involved: `libm` is linked into every program
already, so nothing has to be distributed alongside the compiler and no extra
library has to be found on the library path.

The prototypes are guarded by `__STDC_WANT_IEC_60559_TYPES_EXT__`, which the
build sets. Both the scalar and the complex entry points are required, and
they are gated separately in glibc, so the CMake configuration checks them by
compiling a program that takes the address of one of each. The check fails at
configuration time, naming what is missing, on a libc that exports the symbols
but does not declare them.

## `-DFLANG_RUNTIME_F128_MATH_LIB=libquadmath`

This builds `libflang_rt.quadmath` with unresolved references to GCC's
`libquadmath` library. A Flang driver built with this option will
automatically link `libflang_rt.quadmath` and `libquadmath` libraries
to any Fortran program. This implies that `libquadmath` library
has to be available in the standard library paths, so that linker
can find it. The `libquadmath` library installation into Flang project
distribution is not automatic in CMake currently.

Testing shows that `libquadmath` versions before GCC-9.3.0 have
accuracy issues, so it is recommended to distribute the Flang
package with later versions of `libquadmath`.

Care must be taken by the distributors of a Flang package built
with `REAL(16)` support via `libquadmath` because of its licensing
under the GNU Library General Public License. Moreover, static linking
of `libquadmath` to the Flang users' programs may imply some
restrictions/requirements. This document is not intended to give
any legal advice on distributing such a Flang compiler. Where the `libm`
option above is available, it avoids the question entirely.

## Targets where `long double` is already binary128

Flang compiler targeting systems with `LDBL_MANT_DIG == 113`
may provide `REAL(16)` math support without a `libquadmath`
dependency, using standard `libc` APIs for the `long double`
data type. It is not recommended to use either of the above CMake options
for building Flang compilers for such targets.

## Constant folding

Whichever library is selected is also the one the compiler folds `REAL(16)`
constant expressions through, so `parameter :: v = sin(x)` and `y = sin(x)`
are evaluated by the same implementation. If the option is left empty,
`REAL(16)` is unsupported in the compiler and no folding library is linked.
