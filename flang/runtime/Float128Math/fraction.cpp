//===-- runtime/Float128Math/fraction.cpp ---------------------------------===//
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
// FRACTION (16.9.80)
F128Type RTDEF(Fraction16)(F128Type x) { return Fraction(x); }
#endif

} // extern "C"
} // namespace Fortran::runtime
