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
#include <cassert>
#include "test_macros.h"
#include "../../range_adaptor_types.h"

template <class Iter>
concept canDecrement = requires(Iter it) { --it; } || requires(Iter it) { it--; };

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
    std::ranges::zip_view v(a, NonBidi{buffer});
    using Iter = std::ranges::iterator_t<decltype(v)>;
    static_assert(!canDecrement<Iter>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
