//===-- RoundingModeUtils.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RoundingModeUtils.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/rounding_mode.h"

#include "hdr/fenv_macros.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace fputil {
namespace testing {

int get_fe_rounding(RoundingMode mode) {
  switch (mode) {
  case RoundingMode::Upward:
    return FE_UPWARD;
  case RoundingMode::Downward:
    return FE_DOWNWARD;
  case RoundingMode::TowardZero:
    return FE_TOWARDZERO;
  case RoundingMode::Nearest:
    return FE_TONEAREST;
  }
  __builtin_unreachable();
}

ForceRoundingMode::ForceRoundingMode(RoundingMode mode) {
  old_rounding_mode = quick_get_round();
  rounding_mode = get_fe_rounding(mode);
  if (old_rounding_mode != rounding_mode) {
    int status = set_round(rounding_mode);
    success = (status == 0);
  } else {
    success = true;
  }
}

ForceRoundingMode::~ForceRoundingMode() {
  if (old_rounding_mode != rounding_mode)
    set_round(old_rounding_mode);
}

} // namespace testing
} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL
