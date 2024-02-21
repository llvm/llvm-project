//===-- runtime/Float128Math/cabs.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math-entries.h"

namespace Fortran::runtime {
extern "C" {

#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
// FIXME: the argument should be CppTypeFor<TypeCategory::Complex, 16>,
// and it should be translated into the underlying library's
// corresponding complex128 type.
CppTypeFor<TypeCategory::Real, 16> RTDEF(CAbsF128)(ComplexF128 x) {
  return CAbs<RTNAME(CAbsF128)>::invoke(x);
}
#endif

} // extern "C"
} // namespace Fortran::runtime
