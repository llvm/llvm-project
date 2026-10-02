//===-- lib/runtime/extrema-value.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MAXVAL, MINVAL, and NORM2 entry points.

#include "extrema.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

CppTypeFor<TypeCategory::Integer, 1> RTDEF(MaxvalInteger1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 1, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(MaxvalInteger2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 2, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(MaxvalInteger4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 4, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(MaxvalInteger8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 8, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(MaxvalInteger16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 16, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif

CppTypeFor<TypeCategory::Unsigned, 1> RTDEF(MaxvalUnsigned1)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 1, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Unsigned, 2> RTDEF(MaxvalUnsigned2)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 2, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Unsigned, 4> RTDEF(MaxvalUnsigned4)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 4, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Unsigned, 8> RTDEF(MaxvalUnsigned8)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 8, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Unsigned, 16> RTDEF(MaxvalUnsigned16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 16, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif

// TODO: REAL(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTDEF(MaxvalReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 4, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(MaxvalReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 8, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDEF(MaxvalReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 10, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(MaxvalReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 16, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif

void RTDEF(MaxvalCharacter)(Descriptor &result, const Descriptor &x,
    const char *source, int line, const Descriptor *mask) {
  CharacterMaxOrMin<true>(result, x, 0, source, line, mask, "MAXVAL");
}

CppTypeFor<TypeCategory::Integer, 1> RTDEF(MinvalInteger1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 1, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(MinvalInteger2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 2, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(MinvalInteger4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 4, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(MinvalInteger8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 8, false>(
      x, source, line, dim, mask, "MINVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(MinvalInteger16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 16, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif

CppTypeFor<TypeCategory::Unsigned, 1> RTDEF(MinvalUnsigned1)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 1, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Unsigned, 2> RTDEF(MinvalUnsigned2)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 2, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Unsigned, 4> RTDEF(MinvalUnsigned4)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 4, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Unsigned, 8> RTDEF(MinvalUnsigned8)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 8, false>(
      x, source, line, dim, mask, "MINVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Unsigned, 16> RTDEF(MinvalUnsigned16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 16, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif

// TODO: REAL(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTDEF(MinvalReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 4, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(MinvalReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 8, false>(
      x, source, line, dim, mask, "MINVAL");
}
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDEF(MinvalReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 10, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(MinvalReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 16, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif

void RTDEF(MinvalCharacter)(Descriptor &result, const Descriptor &x,
    const char *source, int line, const Descriptor *mask) {
  CharacterMaxOrMin<false>(result, x, 0, source, line, mask, "MINVAL");
}

void RTDEF(MaxvalDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  if (x.type().IsCharacter()) {
    CharacterMaxOrMin<true>(result, x, dim, source, line, mask, "MAXVAL");
  } else {
    NumericMaxOrMin<true>(result, x, dim, source, line, mask, "MAXVAL");
  }
}
void RTDEF(MinvalDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  if (x.type().IsCharacter()) {
    CharacterMaxOrMin<false>(result, x, dim, source, line, mask, "MINVAL");
  } else {
    NumericMaxOrMin<false>(result, x, dim, source, line, mask, "MINVAL");
  }
}

RT_EXT_API_GROUP_END
} // extern "C"

// NORM2

extern "C" {
RT_EXT_API_GROUP_BEGIN

// TODO: REAL(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTDEF(Norm2_4)(
    const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalReduction<TypeCategory::Real, 4>(
      x, source, line, dim, nullptr, Norm2Accumulator<4>{x}, "NORM2");
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(Norm2_8)(
    const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalReduction<TypeCategory::Real, 8>(
      x, source, line, dim, nullptr, Norm2Accumulator<8>{x}, "NORM2");
}
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDEF(Norm2_10)(
    const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalReduction<TypeCategory::Real, 10>(
      x, source, line, dim, nullptr, Norm2Accumulator<10>{x}, "NORM2");
}
#endif

void RTDEF(Norm2Dim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type);
  if (type->first == TypeCategory::Real) {
    ApplyFloatingPointKind<Norm2Helper, void, true>(
        type->second, terminator, result, x, dim, nullptr, terminator);
  } else {
    terminator.Crash("NORM2: bad type code %d", x.type().raw());
  }
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
