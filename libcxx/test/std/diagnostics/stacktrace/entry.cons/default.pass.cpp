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
    constexpr stacktrace_entry() noexcept;
}
*/

#include <cassert>
#include <stacktrace>
#include <type_traits>

int main(int, char**) {
  static_assert(std::is_nothrow_default_constructible_v<std::stacktrace_entry>);

  std::stacktrace_entry entry;
  // "Postconditions: *this is empty."
  assert(!entry);

  return 0;
}
