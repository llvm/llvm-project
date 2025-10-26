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
  (19.6.3.2) Constructors [stacktrace.entry.cons]

namespace std {
  class stacktrace_entry {
    // [stacktrace.entry.cons], constructors
    constexpr stacktrace_entry(const stacktrace_entry& other) noexcept;
}
*/

#include <cassert>
#include <stacktrace>
#include <type_traits>

#include "test_macros.h"

TEST_NO_TAIL_CALLS
int main(int, char**) {
  static_assert(std::is_nothrow_copy_constructible_v<std::stacktrace_entry>);

  auto e1 = std::stacktrace::current()[0];
  static_assert(noexcept(std::stacktrace_entry(e1)));
  std::stacktrace_entry e2(e1);
  assert(e2 == e1);

  return 0;
}
