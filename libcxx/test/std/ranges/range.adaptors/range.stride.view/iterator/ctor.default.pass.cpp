//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// iterator() requires default_initializable<iterator_t<Base>> = default;

#include <ranges>
#include <type_traits>

#include "../types.h"

struct NonDefaultConstructibleIterator : InputIter<NonDefaultConstructibleIterator> {
  NonDefaultConstructibleIterator() = delete;
  constexpr NonDefaultConstructibleIterator(int) {}
};

struct ViewWithNonDefaultConstructibleIterator : std::ranges::view_base {
  constexpr NonDefaultConstructibleIterator begin() const { return NonDefaultConstructibleIterator{5}; }
  constexpr std::default_sentinel_t end() const { return {}; }
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<ViewWithNonDefaultConstructibleIterator> = true;

// The stride_view iterator is default-constructible iff the iterator type of the range being
// strided is default-constructible.
static_assert(!std::is_default_constructible< std::ranges::iterator_t<ViewWithNonDefaultConstructibleIterator>>());
static_assert(std::is_default_constructible<
              std::ranges::iterator_t< std::ranges::stride_view<std::ranges::ref_view<const int[3]>>>>());

constexpr bool test() {
  {
    // Default construct an iterator over a default-constructible base.
    using SV = std::ranges::stride_view<std::ranges::ref_view<const int[3]>>;
    using It = std::ranges::iterator_t<SV>;
    [[maybe_unused]] It it{};
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
