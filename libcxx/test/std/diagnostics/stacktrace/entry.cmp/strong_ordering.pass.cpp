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
    friend constexpr strong_ordering operator<=>(const stacktrace_entry& x,
                                                 const stacktrace_entry& y) noexcept;
*/

#include <cassert>
#include <cstdint>
#include <stacktrace>
#include "test_macros.h"

namespace {
TEST_NO_TAIL_CALLS TEST_NOINLINE auto func1() { return std::stacktrace::current()[0]; }
TEST_NO_TAIL_CALLS TEST_NOINLINE auto func2() { return std::stacktrace::current()[0]; }
} // namespace

int main(int, char**) {
  std::stacktrace_entry entry1a = func1();
  std::stacktrace_entry entry1b = func1();
  static_assert(noexcept(entry1a <=> entry1b));

  assert(std::strong_ordering::equal == (entry1a <=> entry1b));
  assert(std::strong_ordering::equivalent == (entry1a <=> entry1b));

  std::stacktrace_entry entry2 = func2();

  uintptr_t func1p = uintptr_t(&func1);
  uintptr_t func2p = uintptr_t(&func2);
  if (func1p < func2p) {
    assert(std::strong_ordering::less == (entry1a <=> entry2));
    assert(std::strong_ordering::greater == (entry2 <=> entry1a));
  } else {
    assert(std::strong_ordering::less == (entry2 <=> entry1a));
    assert(std::strong_ordering::greater == (entry1a <=> entry2));
  }

  return 0;
}
