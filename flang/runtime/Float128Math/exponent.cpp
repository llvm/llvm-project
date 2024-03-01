//===-- runtime/Float128Math/exponent.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math-entries.h"
#include "numeric-template-specs.h"

namespace Fortran::runtime {
extern "C" {

#if LDBL_MANT_DIG != 113 && HAS_FLOAT128
// EXPONENT (16.9.75)
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Exponent16_4)(F128Type x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Exponent16_8)(F128Type x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#endif

} // extern "C"
} // namespace Fortran::runtime
