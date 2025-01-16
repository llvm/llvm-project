//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <experimental/stacktrace>

#include <cassert>
#include <concepts>
#include <cstddef>

/*
  (19.6.3) Class stacktrace_entry         [stacktrace.entry]
  (19.6.3.1) Overview                     [stacktrace.entry.overview]

namespace std {
  class stacktrace_entry {
  public:
    using native_handle_type = implementation-defined;                              // [T1] [entry.pass.cpp]

    // [stacktrace.entry.cons], constructors
    constexpr stacktrace_entry() noexcept;                                          // [T2] [entry.cons.pass.cpp]
    constexpr stacktrace_entry(const stacktrace_entry& other) noexcept;             // [T3]
    constexpr stacktrace_entry& operator=(const stacktrace_entry& other) noexcept;  // [T4]
    ~stacktrace_entry();                                                            // [T5]

    // [stacktrace.entry.obs], observers
    constexpr native_handle_type native_handle() const noexcept;                    // [T6] [entry.obs.pass.cpp]
    constexpr explicit operator bool() const noexcept;                              // [T7]

    // [stacktrace.entry.query], query
    string description() const;                                                     // [T8] [entry.query.pass.cpp]
    string source_file() const;                                                     // [T9]
    uint_least32_t source_line() const;                                             // [T10]

    // [stacktrace.entry.cmp], comparison
    friend constexpr bool operator==(const stacktrace_entry& x,
                                     const stacktrace_entry& y) noexcept;           // [T11] [entry.cmp.pass.cpp]
    friend constexpr strong_ordering operator<=>(const stacktrace_entry& x,
                                                 const stacktrace_entry& y) noexcept;  // [T12]
  };
}
*/

int main(int, char**) {
  // [stacktrace.entry.overview]
  // "The class stacktrace_entry models regular ([concepts.object])
  // and three_way_comparable<strong_ordering> ([cmp.concept])."
  static_assert(std::regular<std::stacktrace_entry>);
  static_assert(std::three_way_comparable<std::stacktrace_entry, std::strong_ordering>);

  // [T1]
  // using native_handle_type = implementation-defined;
  //
  // [Implementation note: For our purposes anything that unique identifies this stacktrace_entry
  // would be good enough.  We'll use a pointer-sized numeric type (to represent location of the
  // calling instruction).  Internally this is defined as `uintptr_t`.]
  static_assert(sizeof(std::stacktrace_entry::native_handle_type) >= sizeof(void*));

  return 0;
}
