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
convertRoundingModeToStr(llvm::RoundingMode UseRounding) {
  std::optional<llvm::StringRef> RoundingStr;
  switch (UseRounding) {
  case llvm::RoundingMode::Dynamic:
    RoundingStr = "round.dynamic";
    break;
  case llvm::RoundingMode::NearestTiesToEven:
    RoundingStr = "round.tonearest";
    break;
  case llvm::RoundingMode::NearestTiesToAway:
    RoundingStr = "round.tonearestaway";
    break;
  case llvm::RoundingMode::TowardNegative:
    RoundingStr = "round.downward";
    break;
  case llvm::RoundingMode::TowardPositive:
    RoundingStr = "round.upward";
    break;
  case llvm::RoundingMode::TowardZero:
    RoundingStr = "round.towardZero";
    break;
  default:
    break;
  }
  return RoundingStr;
}

std::optional<llvm::StringRef>
convertExceptionBehaviorToStr(fp::ExceptionBehavior UseExcept) {
  std::optional<llvm::StringRef> ExceptStr;
  switch (UseExcept) {
  case fp::ebStrict:
    ExceptStr = "fpexcept.strict";
    break;
  case fp::ebIgnore:
    ExceptStr = "fpexcept.ignore";
    break;
  case fp::ebMayTrap:
    ExceptStr = "fpexcept.maytrap";
    break;
  }
  return ExceptStr;
}

} // namespace cir
