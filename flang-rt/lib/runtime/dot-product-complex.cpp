//===-- lib/runtime/dot-product-complex.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Complex DOT_PRODUCT entry points.

#include "dot-product.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(CppDotProductComplex4)(CppTypeFor<TypeCategory::Complex, 4> &result,
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  result = DotProduct<TypeCategory::Complex, 4>{}(x, y, source, line);
}
void RTDEF(CppDotProductComplex8)(CppTypeFor<TypeCategory::Complex, 8> &result,
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  result = DotProduct<TypeCategory::Complex, 8>{}(x, y, source, line);
}
#if HAS_FLOAT80
void RTDEF(CppDotProductComplex10)(
    CppTypeFor<TypeCategory::Complex, 10> &result, const Descriptor &x,
    const Descriptor &y, const char *source, int line) {
  result = DotProduct<TypeCategory::Complex, 10>{}(x, y, source, line);
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(CppDotProductComplex16)(
    CppTypeFor<TypeCategory::Complex, 16> &result, const Descriptor &x,
    const Descriptor &y, const char *source, int line) {
  result = DotProduct<TypeCategory::Complex, 16>{}(x, y, source, line);
}
#endif

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
