//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  constexpr iterator& operator+=(difference_type x) requires random_access_range<Base>;
//  constexpr iterator& operator-=(difference_type x) requires random_access_range<Base>;
//  friend constexpr iterator operator+(const iterator& i, difference_type n)
//    requires random_access_range<Base>;
//  friend constexpr iterator operator+(difference_type n, const iterator& i)
//    requires random_access_range<Base>;
//  friend constexpr iterator operator-(const iterator& i, difference_type n)
//    requires random_access_range<Base>;
//  friend constexpr difference_type operator-(const iterator& x, const iterator& y)
//    requires sized_sentinel_for<ziperator<Const>, ziperator<Const>>;

#include <ranges>

#include <array>
#include <concepts>
#include <functional>

#include "../types.h"

template <class T, class U>
concept canPlusEqual = requires(T& t, U& u) { t += u; };

template <class T, class U>
concept canPlus = requires(T& t, U& u) { t + u; };

template <class T, class U>
concept canMinusEqual = requires(T& t, U& u) { t -= u; };

template <class T, class U>
concept canMinus = requires(T& t, U& u) { t - u; };

constexpr bool test() {
  int buffer1[5] = {1, 2, 3, 4, 5};
  SizedRandomAccessView a{buffer1};
  static_assert(std::ranges::random_access_range<decltype(a)>);

  std::array b{4.1, 3.2, 4.3, 0.1, 0.2};
  static_assert(std::ranges::contiguous_range<decltype(b)>);

  {
    // operator+(x, n) and operator+=
    std::ranges::zip_transform_view v(MakeTuple{}, a, b);
    auto it1   = v.begin();
    using Iter = decltype(it1);

    std::same_as<Iter> decltype(auto) it2 = it1 + 3;
    assert(*it2 == std::tuple(4, 0.1));

    std::same_as<Iter> decltype(auto) it3 = 3 + it1;
    assert(*it3 == std::tuple(4, 0.1));

    std::same_as<Iter&> decltype(auto) it1_ref = it1 += 3;
    assert(&it1_ref == &it1);
    assert(*it1_ref == std::tuple(4, 0.1));
    assert(*it1 == std::tuple(4, 0.1));

    static_assert(canPlus<Iter, std::intptr_t>);
    static_assert(canPlusEqual<Iter, std::intptr_t>);
  }

  {
    // operator-(x, n) and operator-=
    std::ranges::zip_transform_view v(MakeTuple{}, a, b);
    auto it1   = v.end();
    using Iter = decltype(it1);

    std::same_as<Iter> decltype(auto) it2 = it1 - 3;
    assert(*it2 == std::tuple(3, 4.3));

    std::same_as<Iter&> decltype(auto) it1_ref = it1 -= 3;
    assert(&it1_ref == &it1);
    assert(*it1_ref == std::tuple(3, 4.3));
    assert(*it1 == std::tuple(3, 4.3));

    static_assert(canMinusEqual<Iter, std::intptr_t>);
    static_assert(canMinus<Iter, std::intptr_t>);
  }

  {
    // operator-(x, y)
    std::ranges::zip_transform_view v(MakeTuple{}, a, b);
    assert((v.end() - v.begin()) == 5);

    auto it1 = v.begin() + 2;
    auto it2 = v.end() - 1;

    using Iter = decltype(it1);

    std::same_as<std::iter_difference_t<Iter>> decltype(auto) n = it1 - it2;
    assert(n == -2);
  }

  {
    // One of the ranges is not random access
    std::ranges::zip_transform_view v(MakeTuple{}, a, b, ForwardSizedView{buffer1});
    auto it1   = v.begin();
    using Iter = decltype(it1);
    static_assert(!canPlus<Iter, std::intptr_t>);
    static_assert(!canPlus<std::intptr_t, Iter>);
    static_assert(!canPlusEqual<Iter, std::intptr_t>);
    static_assert(!canMinus<Iter, std::intptr_t>);
    static_assert(canMinus<Iter, Iter>);
    static_assert(!canMinusEqual<Iter, std::intptr_t>);

    auto it2 = ++v.begin();
    assert((it2 - it1) == 1);
  }

  {
    // One of the ranges does not have sized sentinel
    std::ranges::zip_transform_view v(MakeTuple{}, a, b, InputCommonView{buffer1});
    using Iter = decltype(v.begin());
    static_assert(!canPlus<Iter, std::intptr_t>);
    static_assert(!canPlus<std::intptr_t, Iter>);
    static_assert(!canPlusEqual<Iter, std::intptr_t>);
    static_assert(!canMinus<Iter, std::intptr_t>);
    static_assert(!canMinus<Iter, Iter>);
    static_assert(!canMinusEqual<Iter, std::intptr_t>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
