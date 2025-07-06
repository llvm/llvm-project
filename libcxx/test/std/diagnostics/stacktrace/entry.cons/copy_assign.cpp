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

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) noexcept {
  static_assert(std::is_nothrow_copy_assignable_v<std::stacktrace_entry>);
  std::stacktrace_entry entry_t2;
  std::stacktrace_entry entry_t4;
  entry_t4 = entry_t2;

  return 0;
}
