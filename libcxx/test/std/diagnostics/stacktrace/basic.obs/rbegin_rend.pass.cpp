//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// XFAIL: availability-stacktrace-missing

/*
  (19.6.4.3) Observers [stacktrace.basic.obs]

  const_reverse_iterator rbegin() const noexcept;
  const_reverse_iterator rend() const noexcept;
*/

#include <cassert>
#include <iterator>
#include <stacktrace>

#include "test_macros.h"

_LIBCPP_NOINLINE TEST_NO_TAIL_CALLS std::stacktrace test1() { return std::stacktrace::current(0, 4); }
_LIBCPP_NOINLINE TEST_NO_TAIL_CALLS std::stacktrace test2() { return test1(); }
_LIBCPP_NOINLINE TEST_NO_TAIL_CALLS std::stacktrace test3() { return test2(); }

TEST_NO_TAIL_CALLS
int main(int, char**) {
  std::stacktrace st;
  static_assert(noexcept(st.rbegin()));
  static_assert(noexcept(st.rend()));
  static_assert(std::random_access_iterator<decltype(st.rbegin())>);
  assert(st.rbegin() == st.rend());
  // no longer empty:
  st      = test3();
  auto it = st.rbegin();
  assert(it != st.rend());

  return 0;
}
