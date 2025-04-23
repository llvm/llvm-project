//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -g

#include <cassert>
#include <stacktrace>
#include <type_traits>

/*
  (19.6.3.2) Constructors [stacktrace.entry.cons]

namespace std {
  class stacktrace_entry {
  public:
    // [stacktrace.entry.cons], constructors
    constexpr stacktrace_entry() noexcept;                                          // [T2]
    constexpr stacktrace_entry(const stacktrace_entry& other) noexcept;             // [T3]
    constexpr stacktrace_entry& operator=(const stacktrace_entry& other) noexcept;  // [T4]
    ~stacktrace_entry();                                                            // [T5]

    [. . .]
  };
}
*/

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  // [T2]
  // constexpr stacktrace_entry() noexcept;
  // "Postconditions: *this is empty."
  static_assert(std::is_default_constructible_v<std::stacktrace_entry>);
  static_assert(std::is_nothrow_default_constructible_v<std::stacktrace_entry>);
  std::stacktrace_entry entry_t2;
  assert(!entry_t2);

  // [T3]
  // constexpr stacktrace_entry(const stacktrace_entry& other) noexcept;
  static_assert(std::is_nothrow_copy_constructible_v<std::stacktrace_entry>);
  std::stacktrace_entry entry_t3(entry_t2);

  // [T4]
  // constexpr stacktrace_entry& operator=(const stacktrace_entry& other) noexcept;
  static_assert(std::is_nothrow_copy_assignable_v<std::stacktrace_entry>);
  std::stacktrace_entry entry_t4;
  entry_t4 = entry_t2;

  // [T5]
  // ~stacktrace_entry();
  std::stacktrace_entry* entry_ptr{nullptr};
  delete entry_ptr;
  {
    auto entry_t5(entry_t4); /* construct and immediately let it go out of scope */
  }

  return 0;
}
