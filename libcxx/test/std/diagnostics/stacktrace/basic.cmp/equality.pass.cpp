//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

// (19.6.4.4) Comparisons [stacktrace.basic.cmp]
//
//   template<class Allocator2>
//   friend bool operator==(const basic_stacktrace& x,
//                          const basic_stacktrace<Allocator2>& y) noexcept;

#include <cassert>
#include <stacktrace>
#include "test_macros.h"

// Call chain is: main -> c -> b -> a -> stacktrace::current;
// we're only checking a, b, c in the returned stacktrace, so use max_depth of 3.
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace a(size_t skip = 0) { return std::stacktrace::current(skip, 3); }
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace b(size_t skip = 0) { return a(skip); }
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace c(size_t skip = 0) { return b(skip); }

int main(int, char**) {
  std::stacktrace st0;
  static_assert(noexcept(st0 == st0));
  static_assert(noexcept(st0 != st0));
  assert(st0 == st0);

  std::stacktrace st1 = a();
  assert(st1 != st0);

  std::stacktrace st2 = b();
  assert(st2 != st1);
  assert(st2 != st0);

  std::stacktrace st3 = c();
  assert(st3 != st0);
  assert(st3 != st1);
  assert(st3 != st2);
  assert(c() == st3);

  return 0;
}
