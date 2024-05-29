<!--===- docs/Real16MathSupport.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Flang support for REAL(16) math intrinsics

To support most `REAL(16)` (i.e. 128-bit float) math intrinsics Flang relies
on third-party libraries providing the implementation.

`-DFLANG_RUNTIME_F128_MATH_LIB=libquadmath` CMake option can be used
to build `FortranFloat128Math` library that has unresolved references
to GCC `libquadmath` library. A Flang driver built with this option
will automatically link `FortranFloat128Math` and `libquadmath` libraries
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
any legal advice on distributing such a Flang compiler.

Flang compiler targeting systems with `LDBL_MANT_DIG == 113`
may provide `REAL(16)` math support without a `libquadmath`
dependency, using standard `libc` APIs for the `long double`
data type. It is not recommended to use the above CMake option
for building Flang compilers for such targets.
