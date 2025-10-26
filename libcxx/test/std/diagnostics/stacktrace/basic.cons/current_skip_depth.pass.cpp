//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: no-localization
// XFAIL: availability-stacktrace-missing

/*
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  static basic_stacktrace current(size_type skip, size_type max_depth,
                                  const allocator_type& alloc = allocator_type()) noexcept;
*/

#include <__config_site>
#if _LIBCPP_HAS_LOCALIZATION

#  include <cassert>
#  include <iostream>
#  include <stacktrace>
#  include "test_macros.h"

_LIBCPP_NOINLINE TEST_NO_TAIL_CALLS void test_current_with_skip_depth() {
  // current stack is: [this function, main, (possibly something else, such as libc _start)]
  // so it's probably 3 functions deep -- but certainly at least 2 deep.
  std::stacktrace_entry entry;
  {
    auto st1 = std::stacktrace::current(0);
    std::cout << st1 << '\n';
    assert(st1.size() >= 2);
    auto it1 = st1.begin();
    ++it1;
    entry = *it1; // represents our caller, `main`
  }

  // get current trace again, but skip the 1st
  auto st2 = std::stacktrace::current(1);
  std::cout << st2 << '\n';
  assert(st2.size() >= 1);
  auto it2 = st2.begin();
  assert(*it2 == entry);
}

TEST_NO_TAIL_CALLS
int main(int, char**) {
  static_assert(noexcept(std::stacktrace::current(0, 0)));
  test_current_with_skip_depth();
  return 0;
}

#else
int main() { return 0; }
#endif // _LIBCPP_HAS_LOCALIZATION
