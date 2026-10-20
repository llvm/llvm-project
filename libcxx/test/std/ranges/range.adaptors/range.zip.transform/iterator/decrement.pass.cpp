//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator--() requires bidirectional_range<Base>;
// constexpr iterator operator--(int) requires bidirectional_range<Base>;

#include <array>
#include <cassert>
#include <ranges>

#include "../types.h"

template <class Iter>
concept canDecrement = requires(Iter it) { --it; } || requires(Iter it) { it--; };

constexpr bool test() {
  std::array a{1, 2, 3, 4};
  std::array b{4.1, 3.2, 4.3};
  {
    // all random access
    std::ranges::zip_transform_view v(MakeTuple{}, a, b, std::views::iota(0, 5));
    auto it    = v.end();
    using Iter = decltype(it);
    static_assert(canDecrement<Iter>);

    std::same_as<Iter&> decltype(auto) it_ref = --it;
    assert(&it_ref == &it);

    assert(*it == std::tuple(3, 4.3, 2));

    auto original                         = it;
    std::same_as<Iter> decltype(auto) it2 = it--;
    assert(original == it2);
    assert(*it == std::tuple(2, 3.2, 1));
  }

  {
    // all bidi+
    int buffer[2] = {1, 2};

    std::ranges::zip_transform_view v(MakeTuple{}, BidiCommonView{buffer}, std::views::iota(0, 5));
    auto it    = v.begin();
    using Iter = decltype(it);
    static_assert(canDecrement<Iter>);

    ++it;
    ++it;

    std::same_as<Iter&> decltype(auto) it_ref = --it;
    assert(&it_ref == &it);

    assert(it == ++v.begin());
    assert(*it == std::tuple(2, 1));

    auto original                         = it;
    std::same_as<Iter> decltype(auto) it2 = it--;
    assert(original == it2);
    assert(*it == std::tuple(1, 0));
  }

  {
    // non bidi
    int buffer[3] = {4, 5, 6};
    std::ranges::zip_transform_view v(MakeTuple{}, a, InputCommonView{buffer});
    using Iter = std::ranges::iterator_t<decltype(v)>;
    static_assert(!canDecrement<Iter>);
  }

  int buffer[] = {1, 2, 3, 4, 5, 6};

  {
    // one range
    std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    auto it    = v.end();
    using Iter = decltype(it);

    std::same_as<Iter&> decltype(auto) it_ref = --it;
    assert(&it_ref == &it);

    assert(*it == std::tuple(6));

    auto original                         = it;
    std::same_as<Iter> decltype(auto) it2 = it--;
    assert(original == it2);
    assert(*it == std::tuple(5));
  }

  {
    // two ranges
    std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, std::views::iota(0));
    auto it    = v.begin() + 5;
    using Iter = decltype(it);

    std::same_as<Iter&> decltype(auto) it_ref = --it;
    assert(&it_ref == &it);

    assert(*it == 5);

    auto original                         = it;
    std::same_as<Iter> decltype(auto) it2 = it--;
    assert(original == it2);
    assert(*it == 4);
  }

  {
    // three ranges
    std::ranges::zip_transform_view v(Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::single_view(2.));
    auto it    = v.end();
    using Iter = decltype(it);

    std::same_as<Iter&> decltype(auto) it_ref = --it;
    assert(&it_ref == &it);

    assert(*it == std::tuple(1, 1, 2.0));

    ++it;

    auto original                         = it;
    std::same_as<Iter> decltype(auto) it2 = it--;
    assert(original == it2);
    assert(*it == std::tuple(1, 1, 2.0));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
