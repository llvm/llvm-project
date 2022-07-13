//===-- RoundingModeUtils.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RoundingModeUtils.h"

#include <fenv.h>

namespace __llvm_libc {
namespace testutils {

int get_fe_rounding(RoundingMode mode) {
  switch (mode) {
  case RoundingMode::Upward:
    return FE_UPWARD;
    break;
  case RoundingMode::Downward:
    return FE_DOWNWARD;
    break;
  case RoundingMode::TowardZero:
    return FE_TOWARDZERO;
    break;
  case RoundingMode::Nearest:
    return FE_TONEAREST;
    break;
  }
}

ForceRoundingMode::ForceRoundingMode(RoundingMode mode) {
  old_rounding_mode = fegetround();
  rounding_mode = get_fe_rounding(mode);
  if (old_rounding_mode != rounding_mode)
    fesetround(rounding_mode);
}

ForceRoundingMode::~ForceRoundingMode() {
  if (old_rounding_mode != rounding_mode)
    fesetround(old_rounding_mode);
}

} // namespace testutils
} // namespace __llvm_libc
