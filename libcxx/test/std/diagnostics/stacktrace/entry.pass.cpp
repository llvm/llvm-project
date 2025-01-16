//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// XFAIL: availability-stacktrace-missing

#include <stacktrace>

#include <cassert>
#include <concepts>
#include <cstddef>

/*
  (19.6.3) Class stacktrace_entry         [stacktrace.entry]
  (19.6.3.1) Overview                     [stacktrace.entry.overview]

namespace std {
  class stacktrace_entry {
  public:
    using native_handle_type = implementation-defined;

    // [stacktrace.entry.cons], constructors
    constexpr stacktrace_entry() noexcept;                                          
    constexpr stacktrace_entry(const stacktrace_entry& other) noexcept;             
    constexpr stacktrace_entry& operator=(const stacktrace_entry& other) noexcept;  
    ~stacktrace_entry();                                                            
    
    // [stacktrace.entry.obs], observers
    constexpr native_handle_type native_handle() const noexcept;                    
    constexpr explicit operator bool() const noexcept;                              
    
    // [stacktrace.entry.query], query
    string description() const;                                                     
    string source_file() const;                                                     
    uint_least32_t source_line() const;                                             
    
    // [stacktrace.entry.cmp], comparison
    friend constexpr bool operator==(const stacktrace_entry& x,
                                     const stacktrace_entry& y) noexcept;           
    friend constexpr strong_ordering operator<=>(const stacktrace_entry& x,
                                                 const stacktrace_entry& y) noexcept;
  };
}
*/

int main(int, char**) {
  // [stacktrace.entry.overview]
  // "The class stacktrace_entry models regular ([concepts.object])
  // and three_way_comparable<strong_ordering> ([cmp.concept])."
  static_assert(std::regular<std::stacktrace_entry>);
  static_assert(std::three_way_comparable<std::stacktrace_entry, std::strong_ordering>);

  // using native_handle_type = implementation-defined;
  static_assert(sizeof(std::stacktrace_entry::native_handle_type) >= sizeof(void*));

  return 0;
}
