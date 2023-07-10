//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// To silence a GCC warning-turned-error re. `BadAlloc::value_type`.
// ADDITIONAL_COMPILE_FLAGS: -Wno-unused-local-typedefs

// template<ranges::input_range R,
//          class Allocator = allocator<ranges::range_value_t<R>>>
//   basic_string(from_range_t, R&&, Allocator = Allocator())
//     -> basic_string<ranges::range_value_t<R>, char_traits<ranges::range_value_t<R>>,
//                     Allocator>; // C++23
//
// The deduction guide shall not participate in overload resolution if Allocator
// is a type that does not qualify as an allocator (in addition to the `input_range` concept being satisfied by `R`).

#include <array>
#include <string>

#include "deduction_guides_sfinae_checks.h"
#include "test_allocator.h"

int main(int, char**) {
  using Char = char16_t;

  {
    std::basic_string c(std::from_range, std::array<Char, 0>());
    static_assert(std::is_same_v<decltype(c), std::basic_string<Char>>);
  }

  {
    using Alloc = test_allocator<Char>;
    std::basic_string c(std::from_range, std::array<Char, 0>(), Alloc());
    static_assert(std::is_same_v<decltype(c), std::basic_string<Char, std::char_traits<Char>, Alloc>>);
  }

  // Note: defining `value_type` is a workaround because one of the deduction guides will end up instantiating
  // `basic_string`, and that would fail with a hard error if the given allocator doesn't define `value_type`.
  struct BadAlloc { using value_type = char; };
  SequenceContainerDeductionGuidesSfinaeAway<std::basic_string, std::basic_string<char>, BadAlloc>();

  return 0;
}
