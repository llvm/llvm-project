//===-- runtime/Float128Math/set-exponent.cpp -----------------------------===//
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

#if HAS_LDBL128 || HAS_FLOAT128
// SET_EXPONENT (16.9.171)
F128Type RTDEF(SetExponent16)(F128Type x, std::int64_t p) {
  return SetExponent(x, p);
}
#endif

} // extern "C"
} // namespace Fortran::runtime
