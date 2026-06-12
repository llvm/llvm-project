//===-- lib/runtime/dot-product-real.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Real and Logical DOT_PRODUCT entry points.

#include "dot-product.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

CppTypeFor<TypeCategory::Real, 4> RTDEF(DotProductReal4)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Real, 4>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(DotProductReal8)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Real, 8>{}(x, y, source, line);
}
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDEF(DotProductReal10)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Real, 10>{}(x, y, source, line);
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(DotProductReal16)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Real, 16>{}(x, y, source, line);
}
#endif

bool RTDEF(DotProductLogical)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Logical, 1>{}(x, y, source, line);
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
