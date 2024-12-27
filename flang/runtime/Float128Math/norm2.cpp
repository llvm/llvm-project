//===-- runtime/Float128Math/norm2.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math-entries.h"
#include "numeric-template-specs.h"
#include "reduction-templates.h"

namespace Fortran::runtime {
extern "C" {

#if HAS_LDBL128 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(Norm2_16)(
    const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalReduction<TypeCategory::Real, 16>(
      x, source, line, dim, nullptr, Norm2Accumulator<16>{x}, "NORM2");
}

void RTDEF(Norm2DimReal16)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type);
  RUNTIME_CHECK(
      terminator, type->first == TypeCategory::Real && type->second == 16);
  Norm2Helper<16>{}(result, x, dim, nullptr, terminator);
}
#endif

} // extern "C"
} // namespace Fortran::runtime
