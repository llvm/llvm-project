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

  size_type size() const noexcept;
*/

#include <cassert>
#include <stacktrace>

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test1() { return std::stacktrace::current(0, 4); }
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test2() { return test1(); }
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test3() { return test2(); }

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  std::stacktrace st;
  static_assert(noexcept(st.size()));
  assert(st.size() == 0);
  st = test3();
  assert(st.size() > 0);

  return 0;
}
