//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

#include <stacktrace>

#include <cassert>

/*
  (19.6.3.5) Comparison [stacktrace.entry.cmp]

namespace std {
  class stacktrace_entry {
  public:
    // [stacktrace.entry.cmp], comparison
    friend constexpr bool operator==(const stacktrace_entry& x,
                                     const stacktrace_entry& y) noexcept;           // [T11]
    friend constexpr strong_ordering operator<=>(const stacktrace_entry& x,
                                                 const stacktrace_entry& y) noexcept;  // [T12]

    [. . .]
  };
}
*/

int main(int, char**) {
  // Two empty entries

  std::stacktrace_entry a;
  assert(a.native_handle() == 0);
  assert(!a);

  std::stacktrace_entry b;
  assert(b.native_handle() == 0);
  assert(!b);

  // A non-empty entry.
  std::stacktrace_entry c;
  ((std::__stacktrace::entry*)(&c))->__addr_ = (uintptr_t)&main;
  assert(c);

  // [T11]
  // friend constexpr bool operator==(const stacktrace_entry& x,
  //                                  const stacktrace_entry& y) noexcept;
  // "Returns: true if and only if x and y represent the same
  // stacktrace entry or both x and y are empty."
  // (We represent "empty" with a native_handle of zero.)
  assert(a == b);
  assert(a != c);

  // [T12]
  // friend constexpr strong_ordering operator<=>(const stacktrace_entry& x,
  //                                              const stacktrace_entry& y) noexcept;
  assert(std::strong_ordering::equal == (a <=> b));
  assert(std::strong_ordering::equivalent == (a <=> b));
  assert(std::strong_ordering::greater == (c <=> a));

  return 0;
}
