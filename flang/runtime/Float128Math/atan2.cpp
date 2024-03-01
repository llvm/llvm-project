//===-- runtime/Float128Math/atan2.cpp ------------------------------------===//
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
CppTypeFor<TypeCategory::Real, 16> RTDEF(Atan2F128)(
    CppTypeFor<TypeCategory::Real, 16> x,
    CppTypeFor<TypeCategory::Real, 16> y) {
  return Atan2<RTNAME(Atan2F128)>::invoke(x, y);
}
#endif

} // extern "C"
} // namespace Fortran::runtime
