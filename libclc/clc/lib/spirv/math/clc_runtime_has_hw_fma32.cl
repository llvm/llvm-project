//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/internal/math/clc_runtime_has_hw_fma32.h"

bool __attribute__((noinline)) __clc_runtime_has_hw_fma32(void) {
  return false;
}
