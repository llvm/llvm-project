//===-- random.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/random.h"

#include <time.h>

namespace gwp_asan {
uint32_t getRandomUnsigned32() {
  thread_local uint32_t RandomState = static_cast<uint64_t>(time(nullptr));
  RandomState ^= RandomState << 13;
  RandomState ^= RandomState >> 17;
  RandomState ^= RandomState << 5;
  return RandomState;
}
} // namespace gwp_asan
