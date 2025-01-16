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

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;
*/

#include <cassert>
#include <iterator>
#include <stacktrace>

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test1() { return std::stacktrace::current(0, 4); }
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test2() { return test1(); }
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test3() { return test2(); }

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  std::stacktrace st;
  static_assert(noexcept(st.begin()));
  static_assert(noexcept(st.end()));
  static_assert(std::random_access_iterator<decltype(st.begin())>);
  assert(st.begin() == st.end());
  // no longer empty:
  st = test3();
  assert(!st.empty());
  assert(st.begin() != st.end());
  auto f0 = st[0];
  auto it = st.begin();
  assert(*it == f0);
  assert(it != st.end());

  return 0;
}
