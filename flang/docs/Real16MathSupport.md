<!--===- docs/Real16MathSupport.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Flang support for REAL(16) math intrinsics

To support most `REAL(16)` (i.e. 128-bit float) math intrinsics Flang relies
on third-party libraries providing the implementation.

`-DFLANG_RUNTIME_F128_MATH_LIB=libquadmath` CMake option can be used
to build `libflang_rt.quadmath` library that has unresolved references
to GCC `libquadmath` library. A Flang driver built with this option
will automatically link `libflang_rt.quadmath` and `libquadmath` libraries
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

In addition to the runtime library support described above, Flang may
also consider `REAL(16)` available when the LLVM backend supports the
target's 128-bit floating-point type. This check is independent of the
availability of libraries implementing `REAL(16)` math intrinsics.

As a result, `REAL(16)` variables and arithmetic operations may be accepted
even when no library support for `REAL(16)` math intrinsics is available.
In such cases, references to math intrinsic functions can result in linker
errors rather than frontend diagnostics. For example:

```
FIRModule:(.text+0x97): undefined reference to `_FortranASinF128'
```

In such configurations, basic arithmetic operations such as addition,
subtraction, multiplication, and division may still work if they are
supported by the LLVM backend, while math intrinsics such as `SIN`, `COS`, `EXP`,
and `LOG` will require additional runtime library support.

This distinction can affect programs that use `SELECTED_REAL_KIND` to
determine whether `REAL(16)` is available. For example:

```Fortran
function test(x)
  integer, parameter :: k = merge(16, 4, selected_real_kind(p=33) .eq. 16)
  real(kind=k) :: x
  test = sin(x)
end function
```

When `SELECTED_REAL_KIND(p=33)` is evaluated during constant folding, it may
produce `16` if `REAL(16)` type support is available, even when the
corresponding math intrinsic library support is unavailable. In such cases
the program may compile successfully but fail to link because the required
`REAL(16)` math intrinsic implementations cannot be found.

Users who want to prevent any use of `REAL(16)` regardless of backend
support can use the `-fdisable-real-16` option.
