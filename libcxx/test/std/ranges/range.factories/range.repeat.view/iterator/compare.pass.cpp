//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// repeat_view::<iterator>::operator{==,<=>}

#include <ranges>
#include <cassert>
#include <concepts>

constexpr bool test() {
  // Test unbound
  {
    using R = std::ranges::repeat_view<int>;
    static_assert(std::three_way_comparable<std::ranges::iterator_t<R>>);

    std::ranges::repeat_view<int> r(42);
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    static_assert(std::same_as<decltype(iter1 == iter2), bool>);

    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
    assert(iter2 == iter2);

    assert(!(iter1 < iter1));
    assert(iter1 < iter2);
    assert(!(iter2 < iter1));
    assert(iter1 <= iter1);
    assert(iter1 <= iter2);
    assert(!(iter2 <= iter1));
    assert(!(iter1 > iter1));
    assert(!(iter1 > iter2));
    assert(iter2 > iter1);
    assert(iter1 >= iter1);
    assert(!(iter1 >= iter2));
    assert(iter2 >= iter1);
    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
    assert(iter2 == iter2);
    assert(!(iter1 != iter1));
    assert(iter1 != iter2);
    assert(!(iter2 != iter2));

    assert((iter1 <=> iter2) == std::strong_ordering::less);
    assert((iter1 <=> iter1) == std::strong_ordering::equal);
    assert((iter2 <=> iter1) == std::strong_ordering::greater);

    static_assert(std::same_as<decltype(iter1 <=> iter2), std::strong_ordering>);
  }

  // Test bound
  {
    using R = std::ranges::repeat_view<int, int>;
    static_assert(std::three_way_comparable<std::ranges::iterator_t<R>>);

    std::ranges::repeat_view<int, int> r(42, 10);
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    static_assert(std::same_as<decltype(iter1 == iter2), bool>);

    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
    assert(iter2 == iter2);

    assert(!(iter1 < iter1));
    assert(iter1 < iter2);
    assert(!(iter2 < iter1));
    assert(iter1 <= iter1);
    assert(iter1 <= iter2);
    assert(!(iter2 <= iter1));
    assert(!(iter1 > iter1));
    assert(!(iter1 > iter2));
    assert(iter2 > iter1);
    assert(iter1 >= iter1);
    assert(!(iter1 >= iter2));
    assert(iter2 >= iter1);
    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
    assert(iter2 == iter2);
    assert(!(iter1 != iter1));
    assert(iter1 != iter2);
    assert(!(iter2 != iter2));

    assert((iter1 <=> iter2) == std::strong_ordering::less);
    assert((iter1 <=> iter1) == std::strong_ordering::equal);
    assert((iter2 <=> iter1) == std::strong_ordering::greater);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
