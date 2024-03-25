//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// x += n;
// x + n;
// n + x;
// x -= n;
// x - n;
// x - y;
// All the arithmetic operators have the constraint `requires all-random-access<Const, Views...>;`,
// except `operator-(x, y)` which instead has the constraint
//    `requires (sized_sentinel_for<iterator_t<maybe-const<Const, Views>>,
//                                  iterator_t<maybe-const<Const, Views>>> && ...);`

#include <ranges>

#include <array>
#include <concepts>
#include <functional>

#include "../types.h"

template <class T, class U>
concept canPlusEqual = requires(T& t, U& u) { t += u; };

template <class T, class U>
concept canMinusEqual = requires(T& t, U& u) { t -= u; };

constexpr bool test() {
  int buffer1[5] = {1, 2, 3, 4, 5};
  int buffer2[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  SizedRandomAccessView a{buffer1};
  static_assert(std::ranges::random_access_range<decltype(a)>);
  std::array b{4.1, 3.2, 4.3, 0.1, 0.2};
  static_assert(std::ranges::contiguous_range<decltype(b)>);
  {
    // operator+(x, n) and operator+=
    std::ranges::zip_view v(a, b);
    auto it1 = v.begin();

    auto it2 = it1 + 3;
    auto [x2, y2] = *it2;
    assert(&x2 == &(a[3]));
    assert(&y2 == &(b[3]));

    auto it3 = 3 + it1;
    auto [x3, y3] = *it3;
    assert(&x3 == &(a[3]));
    assert(&y3 == &(b[3]));

    it1 += 3;
    assert(it1 == it2);
    auto [x1, y1] = *it2;
    assert(&x1 == &(a[3]));
    assert(&y1 == &(b[3]));

    using Iter = decltype(it1);
    static_assert(canPlusEqual<Iter, std::intptr_t>);
  }

  {
    // operator-(x, n) and operator-=
    std::ranges::zip_view v(a, b);
    auto it1 = v.end();

    auto it2 = it1 - 3;
    auto [x2, y2] = *it2;
    assert(&x2 == &(a[2]));
    assert(&y2 == &(b[2]));

    it1 -= 3;
    assert(it1 == it2);
    auto [x1, y1] = *it2;
    assert(&x1 == &(a[2]));
    assert(&y1 == &(b[2]));

    using Iter = decltype(it1);
    static_assert(canMinusEqual<Iter, std::intptr_t>);
  }

  {
    // operator-(x, y)
    std::ranges::zip_view v(a, b);
    assert((v.end() - v.begin()) == 5);

    auto it1 = v.begin() + 2;
    auto it2 = v.end() - 1;
    assert((it1 - it2) == -2);
  }

  {
    // in this case sentinel is computed by getting each of the underlying sentinels, so the distance
    // between begin and end for each of the underlying iterators can be different
    std::ranges::zip_view v{ForwardSizedView(buffer1), ForwardSizedView(buffer2)};
    using View = decltype(v);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::random_access_range<View>);

    auto it1 = v.begin();
    auto it2 = v.end();
    // it1 : <buffer1 + 0, buffer2 + 0>
    // it2 : <buffer1 + 5, buffer2 + 9>
    assert((it1 - it2) == -5);
    assert((it2 - it1) == 5);
  }

  {
    // One of the ranges is not random access
    std::ranges::zip_view v(a, b, ForwardSizedView{buffer1});
    using Iter = decltype(v.begin());
    static_assert(!std::invocable<std::plus<>, Iter, std::intptr_t>);
    static_assert(!std::invocable<std::plus<>, std::intptr_t, Iter>);
    static_assert(!canPlusEqual<Iter, std::intptr_t>);
    static_assert(!std::invocable<std::minus<>, Iter, std::intptr_t>);
    static_assert(std::invocable<std::minus<>, Iter, Iter>);
    static_assert(!canMinusEqual<Iter, std::intptr_t>);
  }

  {
    // One of the ranges does not have sized sentinel
    std::ranges::zip_view v(a, b, InputCommonView{buffer1});
    using Iter = decltype(v.begin());
    static_assert(!std::invocable<std::minus<>, Iter, Iter>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
