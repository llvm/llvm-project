//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr __iterator& operator--()

// constexpr __iterator operator--(int)

#include <array>
#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "../../range_adaptor_types.h"

template <class Iter>
concept CanDecrement = requires(Iter it) { --it; } || requires(Iter it) { it--; };

struct NonBidi : IntBufferView {
  using IntBufferView::IntBufferView;
  using iterator = forward_iterator<int*>;
  constexpr iterator begin() const { return iterator(buffer_); }
  constexpr sentinel_wrapper<iterator> end() const { return sentinel_wrapper<iterator>(iterator(buffer_ + size_)); }
};

constexpr bool test() {
  std::array<int, 4> a{1, 2, 3, 4};
  std::array<int, 4> b{5, 6, 7, 8};

  // Test with a single view
  {
    std::ranges::concat_view view(a);
    auto it = std::ranges::next(view.begin(), view.end());
    assert(it == view.end());

    auto& result = --it;
    ASSERT_SAME_TYPE(decltype(result)&, decltype(--it));
    assert(&result == &it);
    assert(result == view.begin() + 3);
  }

  // Test with more than one view
  {
    std::ranges::concat_view view(a, b);
    auto it = std::ranges::next(view.begin(), view.end());
    assert(it == view.end());

    auto& result = --it;
    assert(&result == &it);

    --it;
    assert(*it == 7);
    assert(it == view.begin() + 6);
  }

  // Test going forward and then backward on the same iterator
  {
    std::ranges::concat_view view(a, b);
    auto it = view.begin();
    ++it;
    --it;
    assert(*it == a[0]);
    ++it;
    ++it;
    --it;
    assert(*it == a[1]);
    ++it;
    ++it;
    --it;
    assert(*it == a[2]);
    ++it;
    ++it;
    --it;
    assert(*it == a[3]);
  }

  // Test post-decrement
  {
    std::ranges::concat_view view(a, b);
    auto it = std::ranges::next(view.begin(), view.end());
    assert(it == view.end()); // test the test
    auto result = it--;
    ASSERT_SAME_TYPE(decltype(result), decltype(it--));
    assert(result == view.end());
    assert(it == (result - 1));
  }

  // bidirectional
  {
    int buffer[2] = {1, 2};

    std::ranges::concat_view v(BidiCommonView{buffer}, std::views::iota(0, 5));
    auto it    = v.begin();
    using Iter = decltype(it);

    ++it;
    ++it;

    static_assert(std::is_same_v<decltype(--it), Iter&>);
    auto& it_ref = --it;
    assert(&it_ref == &it);

    assert(it == ++v.begin());

    static_assert(std::is_same_v<decltype(it--), Iter>);
    auto tmp = it--;
    assert(it == v.begin());
    assert(tmp == ++v.begin());
  }

  // non bidirectional
  {
    int buffer[3] = {4, 5, 6};
    std::ranges::concat_view v(a, NonBidi{buffer});
    using Iter = std::ranges::iterator_t<decltype(v)>;
    static_assert(!CanDecrement<Iter>);
  }

  // Cross-boundary decrement: from begin of a later range into the previous range's last element.
  {
    auto v = std::views::concat(a, b);

    auto it = v.begin();
    it += a.size();
    --it;
    assert(*it == a.back());

    auto it2 = v.begin();
    it2 += a.size();
    auto old = it2--;
    static_assert(std::is_same_v<decltype(old), decltype(it2)>);
    assert(*old == b.front());
    assert(*it2 == a.back());
  }

  // Cross-boundary with three ranges
  {
    std::array<int, 3> c{9, 10, 11};
    auto v3 = std::views::concat(a, b, c);

    auto it = v3.begin();
    it += a.size() + b.size();
    --it;
    assert(*it == b.back());

    // const-iterator.
    const auto& cv3 = v3;
    auto cit        = cv3.begin();
    cit += a.size() + b.size();
    --cit;
    assert(*cit == b.back());
  }

  // Cross-boundary decrement where the previous range is empty.
  {
    std::array<int, 0> e{};
    auto v = std::views::concat(a, e, b);

    auto it = v.begin();
    it += a.size();
    --it; // this skips e
    assert(*it == a.back());

    auto it2 = v.begin();
    it2 += a.size();
    auto old = it2--;
    assert(*old == b.front());
    assert(*it2 == a.back());

    // const-iterator
    const auto& cv = v;
    auto cit       = cv.begin();
    cit += a.size();
    --cit;
    assert(*cit == a.back());
  }

  // multiple empty ranges in the middle
  {
    std::array<int, 0> e1{}, e2{};
    auto v = std::views::concat(a, e1, e2, b);

    auto it = v.begin();
    it += a.size();
    --it; // skip e2 and e1
    assert(*it == a.back());
  }

  // Different range types
  {
    std::span<const int> sa{a};
    auto sb = std::ranges::subrange{b.begin(), b.end()};
    std::array<int, 2> c{9, 10};

    auto v = std::views::concat(sa, sb, c);

    auto it = v.begin();
    std::ranges::advance(it, sa.size());
    --it;
    assert(*it == a.back());

    auto it2 = v.begin();
    std::ranges::advance(it2, sa.size() + sb.size());
    --it2;
    assert(*it2 == b.back());

    // const-iterator.
    const auto& cv = v;
    auto cit       = cv.begin();
    std::ranges::advance(cit, sa.size() + sb.size());
    --cit;
    assert(*cit == b.back());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
