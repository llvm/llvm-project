//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>

#include <array>
#include <concepts>
#include <functional>

#include "../../range_adaptor_types.h"

template <class T, class U>
concept canPlusEqual = requires(T& t, U& u) { t += u; };

template <class T, class U>
concept canMinusEqual = requires(T& t, U& u) { t -= u; };

constexpr bool test() {
  int buffer1[5] = {1, 2, 3, 4, 5};
  int buffer2[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  SimpleCommonRandomAccessSized a{buffer1};
  SimpleCommonRandomAccessSized b{buffer2};

  {
    // operator+(x, n) and operator+=
    std::ranges::concat_view v(a, b);
    auto it1 = v.begin();

    auto it2 = it1 + 3;
    auto x2  = *it2;
    assert(x2 == buffer1[3]);

    auto it3 = 3 + it1;
    auto x3  = *it3;
    assert(x3 == buffer1[3]);

    it1 += 3;
    assert(it1 == it2);
    auto x1 = *it2;
    assert(x1 == buffer1[3]);

    using Iter = decltype(it1);
    static_assert(canPlusEqual<Iter, std::intptr_t>);
  }

  {
    // operator-(x, n) and operator-=
    std::ranges::concat_view v(a, b);
    auto it1 = v.end();

    auto it2 = it1 - 3;
    auto x2  = *it2;
    assert(x2 == buffer2[6]);

    it1 -= 3;
    assert(it1 == it2);
    auto x1 = *it2;
    assert(x1 == buffer2[6]);

    using Iter = decltype(it1);
    static_assert(canMinusEqual<Iter, std::intptr_t>);
  }

  {
    // operator-(x, y)
    std::ranges::concat_view v(a, b);
    assert((v.end() - v.begin()) == 14);

    auto it1 = v.begin() + 2;
    auto it2 = v.end() - 1;
    assert((it1 - it2) == -11);
  }

  {
    // One of the ranges is not random access
    std::ranges::concat_view v(a, b, ForwardSizedView{buffer1});
    using Iter = decltype(v.begin());
    static_assert(!std::invocable<std::plus<>, Iter, std::intptr_t>);
    static_assert(!std::invocable<std::plus<>, std::intptr_t, Iter>);
    static_assert(!canPlusEqual<Iter, std::intptr_t>);
    static_assert(!std::invocable<std::minus<>, Iter, std::intptr_t>);
    static_assert(!std::invocable<std::minus<>, Iter, Iter>);
    static_assert(!canMinusEqual<Iter, std::intptr_t>);
  }

  {
    // One of the ranges does not have sized sentinel
    std::ranges::concat_view v(a, b, InputCommonView{buffer1});
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
