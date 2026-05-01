//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// x += n;     x + n;     n + x;
// x -= n;     x - n;     x - y;
//
// All require cartesian-product-is-random-access<...>, except `x - y` which requires
// cartesian-is-sized-sentinel<..., iterator_t, ...>.

#include <array>
#include <cassert>
#include <ranges>

#include "../../range_adaptor_types.h"

template <class T, class U>
concept CanPlusEqual = requires(T& t, const U& u) { t += u; };

template <class T, class U>
concept CanMinusEqual = requires(T& t, const U& u) { t -= u; };

constexpr bool test() {
  std::array a{1, 2, 3};
  std::array b{10, 20, 30, 40};

  { // operator+(x, n), operator+(n, x), operator+=
    std::ranges::cartesian_product_view v(a, b);
    auto it1 = v.begin();

    const auto it2      = it1 + 5; // (a[1], b[1])
    const auto [x2, y2] = *it2;
    assert(&x2 == &a[1]);
    assert(&y2 == &b[1]);

    assert((5 + it1) == it2);

    it1 += 5;
    assert(it1 == it2);

    using Iter = decltype(it1);
    static_assert(CanPlusEqual<Iter, std::ptrdiff_t>);
  }

  { // operator-(x, n), operator-=
    std::ranges::cartesian_product_view v(a, b);
    auto it1 = v.end();

    auto it2      = it1 - 5; // total size 12, so `end - 5 = (a[2], b[3]) - 4 = (a[1], b[3])`
    auto [x2, y2] = *it2;
    assert(&x2 == &a[1]);
    assert(&y2 == &b[3]);

    it1 -= 5;
    assert(it1 == it2);

    using Iter = decltype(it1);
    static_assert(CanMinusEqual<Iter, std::ptrdiff_t>);
  }

  { // operator-(x, y) -- Cartesian distance
    std::ranges::cartesian_product_view v(a, b);
    assert((v.end() - v.begin()) == 12);

    auto it1 = v.begin() + 3; // (a[0], b[3])
    auto it2 = v.begin() + 7; // (a[1], b[3])
    assert((it2 - it1) == 4);
    assert((it1 - it2) == -4);
  }

  { // crossing wrap boundaries -- large positive offset
    std::ranges::cartesian_product_view v(a, b);
    auto it1    = v.begin();
    auto it2    = it1 + 11; // last element
    auto [x, y] = *it2;
    assert(&x == &a[2]);
    assert(&y == &b[3]);
  }

  { // negative offset on iterator near begin
    std::ranges::cartesian_product_view v(a, b);
    auto it1    = v.begin() + 6; // (a[1], b[2])
    auto it2    = it1 + (-2);    // (a[1], b[0])
    auto [x, y] = *it2;
    assert(&x == &a[1]);
    assert(&y == &b[0]);
  }

  { // 3-range distance
    std::array c{100, 200};
    std::ranges::cartesian_product_view v(a, b, c); // size = 3*4*2 = 24
    assert(v.end() - v.begin() == 24);
    assert((v.begin() + 24) == v.end());
  }

  { // not random-access (forward+sized) -> +=, -=, +, - (with diff) not available, but x - y is
    std::ranges::cartesian_product_view v(ForwardSizedView{a}, ForwardSizedView{b});
    using Iter = decltype(v.begin());
    static_assert(!std::invocable<std::plus<>, Iter, std::ptrdiff_t>);
    static_assert(!std::invocable<std::plus<>, std::ptrdiff_t, Iter>);
    static_assert(!CanPlusEqual<Iter, std::ptrdiff_t>);
    static_assert(!std::invocable<std::minus<>, Iter, std::ptrdiff_t>);
    static_assert(!CanMinusEqual<Iter, std::ptrdiff_t>);
    // Iterators with sized-sentinels still support iter - iter
    static_assert(std::invocable<std::minus<>, Iter, Iter>);
  }

  { // not sized-sentinel (input range) -> x - y is not available
    std::ranges::cartesian_product_view v(InputCommonView{a}, ForwardSizedView{b});
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
