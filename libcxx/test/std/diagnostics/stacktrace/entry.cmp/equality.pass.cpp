//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

/*
  (19.6.3.5) Comparison [stacktrace.entry.cmp]

namespace std {
  class stacktrace_entry {
  public:
    // [stacktrace.entry.cmp], comparison
    friend constexpr bool operator==(const stacktrace_entry& x,
                                     const stacktrace_entry& y) noexcept;
*/

#include <cassert>
#include <stacktrace>
#include "test_macros.h"

namespace {
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace_entry func1() { return std::stacktrace::current()[0]; }
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace_entry func2() { return std::stacktrace::current()[0]; }
} // namespace

int main(int, char**) {
  auto entry1 = func1();
  auto entry2 = func2();

  static_assert(noexcept(entry1 == entry2));
  static_assert(noexcept(entry1 != entry2));

  assert(entry1 != entry2);
  auto entry2b = func2();
  assert(entry2 == entry2b);

  return 0;
}
