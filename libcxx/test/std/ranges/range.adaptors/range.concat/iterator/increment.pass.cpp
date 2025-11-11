//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr __iterator& operator++()

// constexpr void operator++(int)

#include <array>
#include <cassert>
#include <concepts>
#include <ranges>
#include <type_traits>
#include <utility>

#include "test_iterators.h"
#include "../../range_adaptor_types.h"

struct InputRange : IntBufferView {
  using IntBufferView::IntBufferView;
  using iterator = cpp20_input_iterator<int*>;
  constexpr iterator begin() const { return iterator(buffer_); }
  constexpr sentinel_wrapper<iterator> end() const { return sentinel_wrapper<iterator>(iterator(buffer_ + size_)); }
};

constexpr bool test() {
  std::array<int, 4> a{1, 2, 3, 4};
  std::array<double, 4> b{1.1, 2.2, 3.3};

  // one view
  {
    std::ranges::concat_view view(a);
    auto it    = view.begin();
    using Iter = decltype(it);
    static_assert(std::is_same_v<decltype(it++), Iter>);
    static_assert(std::is_same_v<decltype(++it), Iter&>);

    auto& result = ++it;
    assert(&result == &it);
    assert(*result == 2);
  }

  // more than one view
  {
    std::ranges::concat_view view(a, b);
    auto it    = view.begin();
    using Iter = decltype(it);
    static_assert(std::is_same_v<decltype(it++), Iter>);
    static_assert(std::is_same_v<decltype(++it), Iter&>);
    auto& result = ++it;
    assert(&result == &it);
    assert(*result == 2);
  }

  // more than one view
  {
    std::ranges::concat_view view(a, b, std::views::iota(0, 5));
    auto it    = view.begin();
    using Iter = decltype(it);
    static_assert(std::is_same_v<decltype(it++), Iter>);
    static_assert(std::is_same_v<decltype(++it), Iter&>);
    auto& result = ++it;
    assert(&result == &it);
    assert(*result == 2);
  }

  // input range
  {
    int buffer[3] = {4, 5, 6};
    std::ranges::concat_view view(a, InputRange{buffer});
    auto it    = view.begin();
    using Iter = decltype(it);
    static_assert(std::is_same_v<decltype(it++), void>);
    static_assert(std::is_same_v<decltype(++it), Iter&>);
    auto& result = ++it;
    assert(&result == &it);
    assert(*result == 2);
  }

  // Increment an iterator multiple times
  {
    std::ranges::concat_view view(a);

    auto it = view.begin();
    assert(*it == a[0]);

    ++it;
    assert(*it == a[1]);
    ++it;
    assert(*it == a[2]);
    ++it;
    assert(*it == a[3]);
  }

  // Different underlying range types; ++ crosses from end of first into start of second.
  {
    std::span<const int> sa{a};
    auto sb = std::ranges::subrange{b.data(), b.data() + b.size()};
    auto v  = std::views::concat(sa, sb);

    auto it = v.begin();
    for (size_t i = 1; i < a.size(); i++) {
      ++it;
    }
    assert(*it == a.back());

    ++it;
    assert(*it == b.front());

    auto it2 = v.begin();
    for (size_t i = 1; i < a.size(); i++) {
      ++it2;
    }
    auto old = it2++;
    assert(*old == a.back());
    assert(*it2 == b.front());

    // Same with a const-iterator.
    const auto& cv = v;
    auto cit       = cv.begin();
    for (size_t i = 1; i < a.size(); i++) {
      ++cit;
    }
    ++cit;
    assert(*cit == b.front());
  }

  // ++ crosses into next range when that next range is empty.
  {
    std::array<int, 0> e{};
    auto v = std::views::concat(a, e, b);

    auto it = v.begin();
    for (size_t i = 1; i < a.size(); i++) {
      ++it;
    }
    ++it; //  skip e
    assert(*it == b.front());

    auto it2 = v.begin();
    for (size_t i = 1; i < a.size(); i++)
      ++it2;
    auto old = it2++;
    assert(*old == a.back());
    assert(*it2 == b.front());

    // Const-iterator.
    const auto& cv = v;
    auto cit       = cv.begin();
    for (size_t i = 1; i < a.size(); i++)
      ++cit;
    ++cit;
    assert(*cit == b.front());
  }

  // Multiple consecutive empty ranges are skipped on ++.
  {
    std::array<int, 0> e1{}, e2{};
    auto v = std::views::concat(a, e1, e2, b);

    auto it = v.begin();
    for (size_t i = 1; i < a.size(); i++) {
      ++it;
    }

    ++it; // skip e1 and e2
    assert(*it == b.front());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
