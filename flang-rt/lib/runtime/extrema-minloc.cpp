//===-- lib/runtime/extrema-minloc.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MINLOC total reduction entry points.

#include "extrema.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(MinlocCharacter)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  CharacterMaxOrMinLoc<false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocInteger1)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 1, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocInteger2)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 2, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocInteger4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 4, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocInteger8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 8, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#ifdef __SIZEOF_INT128__
void RTDEF(MinlocInteger16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 16, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#endif
void RTDEF(MinlocUnsigned1)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 1, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocUnsigned2)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 2, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocUnsigned4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 4, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocUnsigned8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 8, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#ifdef __SIZEOF_INT128__
void RTDEF(MinlocUnsigned16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 16, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#endif
void RTDEF(MinlocReal4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 4, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocReal8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 8, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#if HAS_FLOAT80
void RTDEF(MinlocReal10)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 10, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(MinlocReal16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 16, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#endif

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
