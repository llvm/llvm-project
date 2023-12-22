//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// expected-no-diagnostics

// __iterator() requires default_initializable<iterator_t<_Base>> = default;

#include <ranges>

#include "../types.h"
#include "test_iterators.h"

struct NonDefaultConstructibleIterator : InputIterBase<NonDefaultConstructibleIterator> {
  NonDefaultConstructibleIterator() = delete;
  constexpr NonDefaultConstructibleIterator(int) {}
};

struct ViewWithNonDefaultConstructibleIterator : std::ranges::view_base {
  constexpr NonDefaultConstructibleIterator begin() const { return NonDefaultConstructibleIterator{5}; }
  constexpr std::default_sentinel_t end() const { return {}; }
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<ViewWithNonDefaultConstructibleIterator> = true;

// If the type of the iterator of the range being strided is non-default
// constructible, then the stride view's iterator should not be default
// constructible, either!
static_assert(!std::is_default_constructible< std::ranges::iterator_t<ViewWithNonDefaultConstructibleIterator>>());
// If the type of the iterator of the range being strided is default
// constructible, then the stride view's iterator should be default
// constructible, too!
static_assert(std::is_default_constructible<
              std::ranges::iterator_t< std::ranges::stride_view<std::ranges::ref_view<const int[3]>>>>());
