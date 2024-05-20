//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing
// XFAIL: !has-64-bit-atomics

#include <atomic>
#include <cassert>

#include "test_macros.h"

void test_85107() {
  if constexpr (sizeof(std::__cxx_contention_t) == 8) {
    // https://github.com/llvm/llvm-project/issues/85107
    // [libc++] atomic_wait uses UL_COMPARE_AND_WAIT when it should use UL_COMPARE_AND_WAIT64 on Darwin
    constexpr std::__cxx_contention_t old_val = 0;
    constexpr std::__cxx_contention_t new_val = old_val + (1l << 32);
    std::__cxx_atomic_contention_t ct(new_val);
    std::__libcpp_atomic_wait(&ct, old_val);
  }
}

int main(int, char**) {
  test_85107();

  return 0;
}
