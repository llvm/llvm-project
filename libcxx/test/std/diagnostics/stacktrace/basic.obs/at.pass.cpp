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

  const_reference at(size_type) const;
*/

#include <cassert>
#include <stacktrace>

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test1() { return std::stacktrace::current(0, 4); }
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test2() { return test1(); }
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test3() { return test2(); }

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  auto st = test3();

  assert(st.at(0));
  assert(st.at(1));
  assert(st.at(2));
  assert(st.at(3));

  auto f0 = st.at(0);
  auto f1 = st.at(1);
  auto f2 = st.at(2);
  auto f3 = st.at(3);

  auto it = st.begin();
  assert(*it++ == f0);
  assert(it != st.end());
  assert(*it++ == f1);
  assert(it != st.end());
  assert(*it++ == f2);
  assert(it != st.end());
  assert(*it++ == f3);

  return 0;
}
