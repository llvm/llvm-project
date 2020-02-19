//===-- random.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/random.h"
#include "gwp_asan/common.h"

#include <time.h>

// Initialised to a magic constant so that an uninitialised GWP-ASan won't
// regenerate its sample counter for as long as possible. The xorshift32()
// algorithm used below results in getRandomUnsigned32(0xff82eb50) ==
// 0xfffffea4.
GWP_ASAN_TLS_INITIAL_EXEC uint32_t RandomState = 0xff82eb50;

namespace gwp_asan {
void initPRNG() {
  RandomState = time(nullptr) + getThreadID();
}

uint32_t getRandomUnsigned32() {
  RandomState ^= RandomState << 13;
  RandomState ^= RandomState >> 17;
  RandomState ^= RandomState << 5;
  return RandomState;
}
} // namespace gwp_asan
