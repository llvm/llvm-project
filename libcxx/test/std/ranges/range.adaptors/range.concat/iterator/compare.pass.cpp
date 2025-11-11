//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// operator==(x,y)
// operator==(x, sentinel)

#include <array>
#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "test_range.h"
#include "../types.h"

template <class Iterator>
constexpr void test() {
  using Sentinel   = sentinel_wrapper<Iterator>;
  using View       = minimal_view<Iterator, Sentinel>;
  using ConcatView = std::ranges::concat_view<View>;

  auto make_concat_view = [](auto begin, auto end) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return ConcatView(std::move(view));
  };

  {
    // test with one view
    std::array<int, 5> array{0, 1, 2, 3, 4};
    ConcatView view                          = make_concat_view(array.data(), array.data() + array.size());
    decltype(auto) it1                       = view.begin();
    decltype(auto) it2                       = view.begin();
    std::same_as<bool> decltype(auto) result = (it1 == it2);
    assert(result);

    ++it1;
    assert(!(it1 == it2));
  }

  {
    // test with more than one view
    constexpr static std::array<int, 3> array1{0, 1, 2};
    constexpr static std::array<int, 3> array2{0, 1, 2};
    constexpr static std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    decltype(auto) it1                       = view.begin();
    decltype(auto) it2                       = view.begin();
    std::same_as<bool> decltype(auto) result = (it1 == it2);
    assert(result);

    ++it2;
    ++it2;
    assert(!(it1 == it2));
    ++it2;
    assert(*it1 == *it2);
  }

  {
    // test with more than one view and iterators are in different range
    std::array<int, 3> array1{0, 1, 2};
    std::array<int, 3> array2{4, 5, 6};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    decltype(auto) it1 = view.begin();
    decltype(auto) it2 = view.begin() + 3;

    assert(it1 != it2);
    assert(*it1 == 0);
    assert(*it2 == 4);
    it1++;
    it2++;
    assert(*it1 == 1);
    assert(*it2 == 5);
  }

  {
    std::array<int, 5> array{0, 1, 2, 3, 4};
    ConcatView view = make_concat_view(array.data(), array.data() + array.size());
    assert(!(view.begin() == view.end()));
  }

  {
    // operator==(x, sentinel)
    std::array<int, 2> array1{1, 2};
    std::array<int, 2> array2{3, 4};
    std::ranges::concat_view v(std::views::all(array1), std::views::all(array2));

    auto it = v.begin();
    assert(!(it == std::default_sentinel_t{}));

    it++;
    it++;
    it++;
    it++;
    assert(it == std::default_sentinel_t{});

    // const-iterator
    const auto& cv = v;
    auto cit       = cv.begin();
    ++cit;
    ++cit;
    ++cit;
    ++cit;
    assert(cit == std::default_sentinel_t{});
  }
}

constexpr bool tests() {
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();
  test<cpp17_input_iterator<int const*>>();
  test<forward_iterator<int const*>>();
  test<bidirectional_iterator<int const*>>();
  test<random_access_iterator<int const*>>();
  test<contiguous_iterator<int const*>>();
  test<int const*>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
