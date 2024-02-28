//===-- runtime/Float128Math/norm2.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math-entries.h"
#include "reduction-templates.h"
#include <cmath>

#if LDBL_MANT_DIG == 113 || HAS_FLOAT128

namespace {
using namespace Fortran::runtime;

using AccumType = Norm2AccumType<16>;

struct ABSTy {
  static AccumType compute(AccumType x) {
    return Sqrt<RTNAME(AbsF128)>::invoke(x);
  }
};

struct SQRTTy {
  static AccumType compute(AccumType x) {
    return Sqrt<RTNAME(SqrtF128)>::invoke(x);
  }
};

using Float128Norm2Accumulator = Norm2Accumulator<16, ABSTy, SQRTTy>;
} // namespace

namespace Fortran::runtime {
extern "C" {

CppTypeFor<TypeCategory::Real, 16> RTDEF(Norm2_16)(
    const Descriptor &x, const char *source, int line, int dim) {
  auto accumulator{::Float128Norm2Accumulator(x)};
  return GetTotalReduction<TypeCategory::Real, 16>(
      x, source, line, dim, nullptr, accumulator, "NORM2");
}

void RTDEF(Norm2DimReal16)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type);
  RUNTIME_CHECK(
      terminator, type->first == TypeCategory::Real && type->second == 16);
  DoMaxMinNorm2<TypeCategory::Real, 16, ::Float128Norm2Accumulator>(
      result, x, dim, nullptr, "NORM2", terminator);
}

} // extern "C"
} // namespace Fortran::runtime

#endif
