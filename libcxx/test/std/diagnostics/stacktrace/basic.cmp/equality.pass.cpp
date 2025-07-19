//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -g

/*
  (19.6.4.4) Comparisons [stacktrace.basic.cmp]

  template<class Allocator2>
  friend bool operator==(const basic_stacktrace& x,
                          const basic_stacktrace<Allocator2>& y) noexcept;
*/

#include <cassert>
#include <stacktrace>

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test1() { return std::stacktrace::current(); }

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test2a() { return test1(); }

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test2b() { return test1(); }

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  auto st1a = test1(); // [test1, main, ...]

  static_assert(noexcept(st1a == st1a));

  assert(st1a == st1a);

  auto st1b = st1a;
  assert(st1a == st1b);

  auto st2a = test2a(); // [test1, test2a, main, ...]
  assert(st1a != st2a);

  std::stacktrace empty; // []
  assert(st1a != empty);
  assert(st2a != empty);

  assert(st2a.size() > st1a.size());
  assert(st1a.size() > empty.size());

  auto st2b = test2b(); // [test1, test2b, main, ...]
  assert(st2a.size() == st2b.size());
  assert(st2a != st2b);

  static_assert(noexcept(st2a == st2b));

  return 0;
}
