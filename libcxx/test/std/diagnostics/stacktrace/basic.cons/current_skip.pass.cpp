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
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  static basic_stacktrace current(size_type skip,
                                  const allocator_type& alloc = allocator_type()) noexcept;   [2]
*/

#include <cassert>
#include <stacktrace>
#include "test_macros.h"

/*
  Let t be a stacktrace as-if obtained via basic_stacktrace::current(alloc). Let n be t.size().
  Returns: A basic_stacktrace object where frames_ is direct-non-list-initialized from arguments
  t.begin() + min(n, skip), t.end(), and alloc, or an empty basic_stacktrace object if the
  initialization of frames_ failed.
*/
_LIBCPP_NOINLINE TEST_NO_TAIL_CALLS void test_current_with_skip() {
  // Use default allocator for simplicity; alloc is covered above
  auto st_skip0 = std::stacktrace::current();
  assert(st_skip0.size() >= 2);
  auto st_skip1 = std::stacktrace::current(1);
  assert(st_skip1.size() >= 1);
  assert(st_skip0.size() == st_skip1.size() + 1);
  assert(st_skip0[1] == st_skip1[0]);
  auto st_skip_many = std::stacktrace::current(1 << 20);
  assert(st_skip_many.empty());
}

TEST_NO_TAIL_CALLS
int main(int, char**) {
  static_assert(noexcept(std::stacktrace::current(0)));
  test_current_with_skip();
  return 0;
}
