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
  (19.6.3.2) Constructors [stacktrace.entry.cons]

namespace std {
  class stacktrace_entry {
    // [stacktrace.entry.cons], constructors
    constexpr stacktrace_entry& operator=(const stacktrace_entry& other) noexcept;
}
*/

#include <cassert>
#include <stacktrace>
#include <type_traits>

int main(int, char**) {
  auto entry1 = std::stacktrace::current()[0];

  static_assert(std::is_nothrow_copy_assignable_v<std::stacktrace_entry>);
  std::stacktrace_entry entry2 = entry1;
  assert(entry2 == entry1);

  return 0;
}
