//===-- lib/runtime/dot-product-integer.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Integer and Unsigned DOT_PRODUCT entry points.

#include "dot-product.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

CppTypeFor<TypeCategory::Integer, 1> RTDEF(DotProductInteger1)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Integer, 1>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(DotProductInteger2)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Integer, 2>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(DotProductInteger4)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Integer, 4>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(DotProductInteger8)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Integer, 8>{}(x, y, source, line);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(DotProductInteger16)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Integer, 16>{}(x, y, source, line);
}
#endif

CppTypeFor<TypeCategory::Unsigned, 1> RTDEF(DotProductUnsigned1)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Unsigned, 1>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Unsigned, 2> RTDEF(DotProductUnsigned2)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Unsigned, 2>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Unsigned, 4> RTDEF(DotProductUnsigned4)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Unsigned, 4>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Unsigned, 8> RTDEF(DotProductUnsigned8)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Unsigned, 8>{}(x, y, source, line);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Unsigned, 16> RTDEF(DotProductUnsigned16)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Unsigned, 16>{}(x, y, source, line);
}
#endif

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
