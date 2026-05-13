//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

// (19.6.4.3) Observers [stacktrace.basic.obs]
//
//   size_type size() const noexcept;

#include <cassert>
#include <iostream>
#include <stacktrace>
#include "test_macros.h"

// Call chain is: main -> c -> b -> a -> stacktrace::current
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace a() { return std::stacktrace::current(); }
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace b() { return a(); }
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace c() { return b(); }

int main(int, char**) {
  std::stacktrace st;
  std::cout << st.size() << '\n';
  std::cout << std::to_string(st) << '\n';

  static_assert(noexcept(st.size()));
  assert(st.size() == 0);

  st = c();
  std::cout << st.size() << '\n';
  std::cout << std::to_string(st) << '\n';
  assert(st.size() >= 4); // at least 4 frames: a, b, c, main

  return 0;
}
