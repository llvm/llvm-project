//===-- lib/quadmath/numeric-template-specs.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_QUADMATH_NUMERIC_TEMPLATE_SPECS_H_
#define FLANG_RT_QUADMATH_NUMERIC_TEMPLATE_SPECS_H_

#include "math-entries.h"
#include "flang-rt/runtime/numeric-templates.h"

namespace Fortran::runtime {
using F128Type = CppTypeFor<TypeCategory::Real, 16>;

template <> struct ABSTy<F128Type> {
  static F128Type compute(F128Type x) { return Abs<true>::invoke(x); }
};

template <> struct FREXPTy<F128Type> {
  static F128Type compute(F128Type x, int *e) {
    return Frexp<true>::invoke(x, e);
  }
};

template <> struct ILOGBTy<F128Type> {
  static int compute(F128Type x) { return Ilogb<true>::invoke(x); }
};

template <> struct ISINFTy<F128Type> {
  static bool compute(F128Type x) { return Isinf<true>::invoke(x); }
};

template <> struct ISNANTy<F128Type> {
  static bool compute(F128Type x) { return Isnan<true>::invoke(x); }
};

template <> struct LDEXPTy<F128Type> {
  template <typename ET> static F128Type compute(F128Type x, ET p) {
    return Ldexp<true>::invoke(x, p);
  }
};

template <> struct QNANTy<F128Type> {
  static F128Type compute() { return F128_RT_QNAN; }
};

template <> struct SQRTTy<F128Type> {
  static F128Type compute(F128Type x) { return Sqrt<true>::invoke(x); }
};

} // namespace Fortran::runtime
#endif // FLANG_RT_QUADMATH_NUMERIC_TEMPLATE_SPECS_H_
