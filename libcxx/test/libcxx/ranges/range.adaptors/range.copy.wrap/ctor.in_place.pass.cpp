//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class ...Args>
// explicit <copyable-box>::<copyable-box>(in_place_t, Args&& ...args);

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility> // in_place_t

#include "types.h"

struct UnknownType { };

template<bool Noexcept>
struct NothrowConstructible {
  explicit NothrowConstructible(int) noexcept(Noexcept);
};

constexpr bool test() {
  // Test the primary template
  {
    using Box = std::ranges::__copyable_box<CopyConstructible>;
    Box x(std::in_place, 5);
    assert((*x).value == 5);

    static_assert(!std::is_constructible_v<Box, std::in_place_t, UnknownType>);
  }

  // Test optimization #1
  {
    using Box = std::ranges::__copyable_box<Copyable>;
    Box x(std::in_place, 5);
    assert((*x).value == 5);

    static_assert(!std::is_constructible_v<Box, std::in_place_t, UnknownType>);
  }

  // Test optimization #2
  {
    using Box = std::ranges::__copyable_box<NothrowCopyConstructible>;
    Box x(std::in_place, 5);
    assert((*x).value == 5);

    static_assert(!std::is_constructible_v<Box, std::in_place_t, UnknownType>);
  }

  static_assert( std::is_nothrow_constructible_v<std::ranges::__copyable_box<NothrowConstructible<true>>, std::in_place_t, int>);
  static_assert(!std::is_nothrow_constructible_v<std::ranges::__copyable_box<NothrowConstructible<false>>, std::in_place_t, int>);

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
