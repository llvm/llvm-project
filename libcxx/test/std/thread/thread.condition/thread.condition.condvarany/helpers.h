//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_THREAD_CONDITION_CONDVARANY_HELPERS_H
#define TEST_STD_THREAD_CONDITION_CONDVARANY_HELPERS_H

#include <chrono>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER >= 17

struct ElapsedTimeCheck {
  ElapsedTimeCheck(std::chrono::steady_clock::duration timeoutDuration)
      : timeout_(std::chrono::steady_clock::now() + timeoutDuration) {}

  ElapsedTimeCheck(ElapsedTimeCheck&&)            = delete;
  ElapsedTimeCheck& operator=(ElapsedTimeCheck&&) = delete;

  ~ElapsedTimeCheck() { assert(std::chrono::steady_clock::now() < timeout_); }

  std::chrono::time_point<std::chrono::steady_clock> timeout_;
};

#endif

#endif // TEST_STD_THREAD_CONDITION_CONDVARANY_HELPERS_H
