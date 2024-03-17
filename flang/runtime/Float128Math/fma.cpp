//===-- runtime/Float128Math/fma.cpp --------------------------------------===//
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
CppTypeFor<TypeCategory::Real, 16> RTDEF(FmaF128)(
    CppTypeFor<TypeCategory::Real, 16> x, CppTypeFor<TypeCategory::Real, 16> y,
    CppTypeFor<TypeCategory::Real, 16> z) {
  return Fma<true>::invoke(x, y, z);
}
#endif

} // extern "C"
} // namespace Fortran::runtime
