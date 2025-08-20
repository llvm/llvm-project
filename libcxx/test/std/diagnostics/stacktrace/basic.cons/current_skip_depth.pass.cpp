//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

/*
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  static basic_stacktrace current(size_type skip, size_type max_depth,
                                  const allocator_type& alloc = allocator_type()) noexcept;
*/

#include <cassert>
#include <iostream>
#include <stacktrace>

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void test_current_with_skip_depth() {
  // current stack is: [this function, main, (possibly something else, e.g. `_start` from libc)]
  // so it's probably 3 functions deep -- but certainly at least 2 deep.
  auto st = std::stacktrace::current(0);
  std::cout << st << '\n';
  assert(st.size() >= 2);
  auto it = st.begin();
  ++it;
  auto entry = *it; // represents our caller, `main`

  // get current trace again, but skip the 1st
  st = std::stacktrace::current(1);
  std::cout << st << '\n';
  assert(st.size() >= 1);
  it = st.begin();
  assert(*it == entry);
}

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  static_assert(noexcept(std::stacktrace::current(0, 0)));
  test_current_with_skip_depth();
  return 0;
}
