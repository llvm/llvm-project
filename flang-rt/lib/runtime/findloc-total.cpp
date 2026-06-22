//===-- lib/runtime/findloc-total.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements total FINDLOC for all required operand types and shapes and
// result integer kinds.

#include "findloc.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(Findloc)(Descriptor &result, const Descriptor &x,
    const Descriptor &target, int kind, const char *source, int line,
    const Descriptor *mask, bool back) {
  int rank{x.rank()};
  SubscriptValue extent[1]{rank};
  result.Establish(TypeCategory::Integer, kind, nullptr, 1, extent,
      CFI_attribute_allocatable);
  result.GetDimension(0).SetBounds(1, extent[0]);
  Terminator terminator{source, line};
  if (int stat{result.Allocate(kNoAsyncObject)}) {
    terminator.Crash(
        "FINDLOC: could not allocate memory for result; STAT=%d", stat);
  }
  CheckIntegerKind(terminator, kind, "FINDLOC");
  auto xType{x.type().GetCategoryAndKind()};
  auto targetType{target.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, xType.has_value() && targetType.has_value());
  switch (xType->first) {
  case TypeCategory::Integer:
  case TypeCategory::Unsigned:
    ApplyIntegerKind<
        TotalNumericFindlocSource<TypeCategory::Integer>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, 0, mask, back, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<
        TotalNumericFindlocSource<TypeCategory::Real>::template Functor, void>(
        xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, 0, mask, back, terminator);
    break;
  case TypeCategory::Complex:
    ApplyFloatingPointKind<
        TotalNumericFindlocSource<TypeCategory::Complex>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, 0, mask, back, terminator);
    break;
  case TypeCategory::Character:
    RUNTIME_CHECK(terminator,
        targetType->first == TypeCategory::Character &&
            targetType->second == xType->second);
    ApplyCharacterKind<CharacterFindlocHelper, void>(xType->second, terminator,
        result, x, target, kind, mask, back, terminator);
    break;
  case TypeCategory::Logical:
    RUNTIME_CHECK(terminator, targetType->first == TypeCategory::Logical);
    LogicalFindlocHelper(result, x, target, kind, mask, back, terminator);
    break;
  default:
    terminator.Crash(
        "FINDLOC: bad data type code (%d) for array", x.type().raw());
  }
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
