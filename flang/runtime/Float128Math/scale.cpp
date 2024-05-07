//===-- runtime/Float128Math/scale.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math-entries.h"
#include "numeric-template-specs.h"
#include <limits>

namespace Fortran::runtime {
extern "C" {

#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
F128Type RTDEF(Scale16)(F128Type x, std::int64_t p) {
  auto ip{static_cast<int>(p)};
  if (ip != p) {
    ip = p < 0 ? std::numeric_limits<int>::min()
               : std::numeric_limits<int>::max();
  }
  return LDEXPTy<F128Type>::compute(x, ip);
}
#endif

} // extern "C"
} // namespace Fortran::runtime
