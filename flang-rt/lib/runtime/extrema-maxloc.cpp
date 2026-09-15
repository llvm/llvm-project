//===-- lib/runtime/extrema-maxloc.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MAXLOC total reduction entry points.

#include "extrema.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(MaxlocCharacter)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  CharacterMaxOrMinLoc<true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocInteger1)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 1, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocInteger2)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 2, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocInteger4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 4, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocInteger8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 8, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#ifdef __SIZEOF_INT128__
void RTDEF(MaxlocInteger16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 16, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#endif
void RTDEF(MaxlocUnsigned1)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 1, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocUnsigned2)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 2, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocUnsigned4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 4, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocUnsigned8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 8, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#ifdef __SIZEOF_INT128__
void RTDEF(MaxlocUnsigned16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 16, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#endif
void RTDEF(MaxlocReal4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 4, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocReal8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 8, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#if HAS_FLOAT80
void RTDEF(MaxlocReal10)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 10, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(MaxlocReal16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 16, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#endif

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
