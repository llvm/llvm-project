//===-- lib/runtime/findloc-dim.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements FINDLOC with DIM= for all required operand types and shapes and
// result integer kinds.

#include "findloc.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(FindlocDim)(Descriptor &result, const Descriptor &x,
    const Descriptor &target, int kind, int dim, const char *source, int line,
    const Descriptor *mask, bool back) {
  Terminator terminator{source, line};
  CheckIntegerKind(terminator, kind, "FINDLOC");
  auto xType{x.type().GetCategoryAndKind()};
  auto targetType{target.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, xType.has_value() && targetType.has_value());
  switch (xType->first) {
  case TypeCategory::Integer:
  case TypeCategory::Unsigned:
    ApplyIntegerKind<
        PartialNumericFindlocSource<TypeCategory::Integer>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<
        PartialNumericFindlocSource<TypeCategory::Real>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Complex:
    ApplyFloatingPointKind<
        PartialNumericFindlocSource<TypeCategory::Complex>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Character:
    RUNTIME_CHECK(terminator,
        targetType->first == TypeCategory::Character &&
            targetType->second == xType->second);
    ApplyCharacterKind<PartialCharacterFindlocHelper, void>(xType->second,
        terminator, result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Logical:
    RUNTIME_CHECK(terminator, targetType->first == TypeCategory::Logical);
    PartialLogicalFindlocHelper(
        result, x, target, kind, dim, mask, back, terminator);
    break;
  default:
    terminator.Crash(
        "FINDLOC: bad data type code (%d) for array", x.type().raw());
  }
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
