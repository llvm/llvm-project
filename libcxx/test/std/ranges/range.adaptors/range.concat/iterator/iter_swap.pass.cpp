//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <ranges>

#include <array>
#include <cassert>
#include <iterator>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template <class It>
concept has_iter_swap = requires (It it) {
  std::ranges::iter_swap(it, it);
};

template <class Iterator, bool IsNoexcept>
constexpr void test() {
  using Sentinel = sentinel_wrapper<Iterator>;
  using View = minimal_view<Iterator, Sentinel>;
  using ConcatView = std::ranges::concat_view<View>;
  using ConcatIterator = std::ranges::iterator_t<ConcatView>;

  auto make_concat_view = [](auto begin, auto end) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return ConcatView(std::move(view));
  };

  {
    std::array<int, 5> array{0,1,2,3,4};
    ConcatView view = make_concat_view(array.data(), array.data() + array.size());
    std::array<int, 5> another_array{5,6,7,8,9};
    ConcatView another_view = make_concat_view(another_array.data(), another_array.data() + another_array.size());
    auto it1 = view.begin();
    auto it2 = another_view.begin();

    static_assert(std::is_same_v<decltype(iter_swap(it1, it2)), void>);
    static_assert(noexcept(iter_swap(it1, it2)) == IsNoexcept);

    assert(*it1 == 0 && *it2 == 5); // test the test
    iter_swap(it1, it2);
    assert(*it1 == 5);
    assert(*it2 == 0);
  }
}

constexpr bool tests() {
   test<cpp17_input_iterator<int*>,           /* noexcept */ false>();
   test<forward_iterator<int*>,               /* noexcept */ false>();
   test<bidirectional_iterator<int*>,         /* noexcept */ false>();
   test<random_access_iterator<int*>,         /* noexcept */ false>();
   test<contiguous_iterator<int*>,            /* noexcept */ false>();
   test<int*,                                 /* noexcept */ false>();

  // Test that iter_swap requires the underlying iterator to be iter_swappable
  {
    using Iterator = int const*;
    using View = minimal_view<Iterator, Iterator>;
    using ConcatView = std::ranges::concat_view<View>;
    using ConcatIterator = std::ranges::iterator_t<ConcatView>;
    static_assert(!std::indirectly_swappable<Iterator>);
    static_assert(!has_iter_swap<ConcatIterator>);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
