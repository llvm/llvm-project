//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// class enumerate_view::iterator

// constexpr const iterator_t<Base>& base() const & noexcept;
// constexpr iterator_t<Base> base() &&;

#include <array>
#include <cassert>
#include <concepts>
#include <memory>
#include <ranges>
#include <utility>
#include <tuple>

#include "test_iterators.h"
#include "../types.h"

template <class Iterator>
constexpr void testBase() {
  using Sentinel          = sentinel_wrapper<Iterator>;
  using View              = MinimalView<Iterator, Sentinel>;
  using EnumerateView     = std::ranges::enumerate_view<View>;
  using EnumerateIterator = std::ranges::iterator_t<EnumerateView>;

  auto make_enumerate_view = [](auto begin, auto end) {
    View view{Iterator(std::to_address(base(begin))), Sentinel(Iterator(std::to_address(base(end))))};

    return EnumerateView(std::move(view));
  };

  std::array array{0, 1, 2, 3, 84};
  const auto view = make_enumerate_view(array.begin(), array.end());

  // Test the const& version
  {
    EnumerateIterator const it                          = view.begin();
    std::same_as<const Iterator&> decltype(auto) result = it.base();
    ASSERT_NOEXCEPT(it.base());
    assert(base(result) == std::to_address(base(array.begin())));
  }

  // Test the && version
  {
    EnumerateIterator it                         = view.begin();
    std::same_as<Iterator> decltype(auto) result = std::move(it).base();
    assert(base(result) == std::to_address(base(array.begin())));
  }
}

constexpr bool test() {
  testBase<cpp17_input_iterator<int*>>();
  testBase<cpp20_input_iterator<int*>>();
  testBase<forward_iterator<int*>>();
  testBase<bidirectional_iterator<int*>>();
  testBase<random_access_iterator<int*>>();
  testBase<contiguous_iterator<int*>>();
  testBase<int*>();
  testBase<int const*>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
