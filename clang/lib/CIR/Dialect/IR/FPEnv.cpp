//===-- FPEnv.cpp ---- FP Environment -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains the implementations of entities that describe floating
/// point environment.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/FPEnv.h"

namespace cir {

std::optional<llvm::StringRef>
convertRoundingModeToStr(llvm::RoundingMode useRounding) {
  std::optional<llvm::StringRef> roundingStr;
  switch (useRounding) {
  case llvm::RoundingMode::Dynamic:
    roundingStr = "round.dynamic";
    break;
  case llvm::RoundingMode::NearestTiesToEven:
    roundingStr = "round.tonearest";
    break;
  case llvm::RoundingMode::NearestTiesToAway:
    roundingStr = "round.tonearestaway";
    break;
  case llvm::RoundingMode::TowardNegative:
    roundingStr = "round.downward";
    break;
  case llvm::RoundingMode::TowardPositive:
    roundingStr = "round.upward";
    break;
  case llvm::RoundingMode::TowardZero:
    roundingStr = "round.towardZero";
    break;
  default:
    break;
  }
  return roundingStr;
}

std::optional<llvm::StringRef>
convertExceptionBehaviorToStr(fp::ExceptionBehavior useExcept) {
  std::optional<llvm::StringRef> exceptStr;
  switch (useExcept) {
  case fp::ebStrict:
    exceptStr = "fpexcept.strict";
    break;
  case fp::ebIgnore:
    exceptStr = "fpexcept.ignore";
    break;
  case fp::ebMayTrap:
    exceptStr = "fpexcept.maytrap";
    break;
  }
  return exceptStr;
}

} // namespace cir
