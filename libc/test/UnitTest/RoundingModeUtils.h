//===-- RoundingModeUtils.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_TESTUTILS_ROUNDINGMODEUTILS_H
#define LLVM_LIBC_UTILS_TESTUTILS_ROUNDINGMODEUTILS_H

#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace fputil {
namespace testing {

enum class RoundingMode : uint8_t { Upward, Downward, TowardZero, Nearest };

struct ForceRoundingMode {
  ForceRoundingMode(RoundingMode);
  ~ForceRoundingMode();

  int old_rounding_mode;
  int rounding_mode;
  bool success;
};

template <RoundingMode R> struct ForceRoundingModeTest : ForceRoundingMode {
  ForceRoundingModeTest() : ForceRoundingMode(R) {}
};

} // namespace testing
} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_UTILS_TESTUTILS_ROUNDINGMODEUTILS_H
