//===-- runtime/Float128Math/y0.cpp ---------------------------------------===//
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
CppTypeFor<TypeCategory::Real, 16> RTDEF(Y0F128)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Y0<RTNAME(Y0F128)>::invoke(x);
}
#endif

} // extern "C"
} // namespace Fortran::runtime
