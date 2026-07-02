//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires forward_range<maybe-const<Const, First>>;

#include <array>
#include <cassert>
#include <ranges>
#include <tuple>

#include "../../range_adaptor_types.h"

// First range may be input-only; second must be forward.
struct InputCommonFirst : IntBufferView {
  using IntBufferView::IntBufferView;
  using iterator = cpp20_input_iterator<int*>;
  constexpr iterator begin() const { return iterator(buffer_); }
  constexpr sentinel_wrapper<iterator> end() const { return sentinel_wrapper<iterator>(iterator(buffer_ + size_)); }
};

constexpr bool test() {
  std::array a{1, 2, 3};
  std::array b{10, 20};

  { // random-access -- operator++ advances the rightmost iterator first, then carries left
    std::ranges::cartesian_product_view v(a, b);
    auto it    = v.begin();
    using Iter = decltype(it);

    assert(*it == std::tuple(1, 10));

    static_assert(std::is_same_v<decltype(++it), Iter&>);
    auto& it_ref = ++it;
    assert(&it_ref == &it);

    assert(*it == std::tuple(1, 20));

    // wrap: rightmost overflows, leftmost advances
    ++it;
    assert(*it == std::tuple(2, 10));

    static_assert(std::is_same_v<decltype(it++), Iter>);
    auto copy = it++;
    assert(*copy == std::tuple(2, 10));
    assert(*it == std::tuple(2, 20));
  }

  { // 3-range wraparound through two levels
    std::array c{100, 200};
    std::ranges::cartesian_product_view v(a, b, c);
    auto it = v.begin();

    // (a[0], b[0], c[0])
    assert(*it == std::tuple(1, 10, 100));
    ++it;
    assert(*it == std::tuple(1, 10, 200));
    ++it; // wrap c, advance b
    assert(*it == std::tuple(1, 20, 100));
    ++it;
    assert(*it == std::tuple(1, 20, 200));
    ++it; // wrap c and b, advance a
    assert(*it == std::tuple(2, 10, 100));
  }

  { // bidirectional first range
    std::ranges::cartesian_product_view v(BidiCommonView{a});
    auto it    = v.begin();
    using Iter = decltype(it);

    static_assert(std::is_same_v<decltype(++it), Iter&>);
    static_assert(std::is_same_v<decltype(it++), Iter>);
    ++it;
    assert(*it == std::tuple(2));
    auto copy = it++;
    assert(*copy == std::tuple(2));
    assert(*it == std::tuple(3));
  }

  { // forward-only first range
    std::ranges::cartesian_product_view v(ForwardSizedView{a});
    auto it    = v.begin();
    using Iter = decltype(it);

    static_assert(std::is_same_v<decltype(++it), Iter&>);
    static_assert(std::is_same_v<decltype(it++), Iter>);
    ++it;
    assert(*it == std::tuple(2));
  }

  { // input-only first range -- operator++(int) returns void
    std::ranges::cartesian_product_view v(InputCommonFirst{a}, BidiCommonView{b});
    auto it    = v.begin();
    using Iter = decltype(it);

    static_assert(std::is_same_v<decltype(++it), Iter&>);
    static_assert(std::is_same_v<decltype(it++), void>);

    assert(std::get<1>(*it) == 10);
    ++it; // wraps b first
    assert(std::get<1>(*it) == 20);
    it++;
    // After this: the inner b wraps, outer (input) advances
    assert(std::get<1>(*it) == 10);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
