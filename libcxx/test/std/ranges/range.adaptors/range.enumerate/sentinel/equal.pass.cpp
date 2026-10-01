//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// class enumerate_view::sentinel

// template<bool OtherConst>
//   requires sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr bool operator==(const iterator<OtherConst>& x, const sentinel& y);

#include <cassert>
#include <compare>
#include <ranges>
#include <tuple>
#include <utility>

#include "../../range_adaptor_types.h"
#include "test_iterators.h"
#include "test_range.h"

using Iterator      = random_access_iterator<int*>;
using ConstIterator = contiguous_iterator<const int*>;

template <bool Const>
struct ComparableSentinel {
  using Iter = std::conditional_t<Const, ConstIterator, Iterator>;
  Iter iter_;

  explicit ComparableSentinel() = default;
  constexpr explicit ComparableSentinel(const Iter& it) : iter_(it) {}

  constexpr friend bool operator==(const Iterator& i, const ComparableSentinel& s) { return base(i) == base(s.iter_); }

  constexpr friend bool operator==(const ConstIterator& i, const ComparableSentinel& s) {
    return base(i) == base(s.iter_);
  }
};

struct ComparableView : IntBufferView {
  using IntBufferView::IntBufferView;

  constexpr auto begin() { return Iterator(buffer_); }
  constexpr auto begin() const { return ConstIterator(buffer_); }
  constexpr auto end() { return ComparableSentinel<false>(Iterator(buffer_ + size_)); }
  constexpr auto end() const { return ComparableSentinel<true>(ConstIterator(buffer_ + size_)); }
};

struct ConstIncompatibleView : IntBufferView {
  using IntBufferView::IntBufferView;

  constexpr random_access_iterator<int*> begin() { return random_access_iterator<int*>(buffer_); }
  constexpr contiguous_iterator<const int*> begin() const { return contiguous_iterator<const int*>(buffer_); }
  constexpr sentinel_wrapper<random_access_iterator<int*>> end() {
    return sentinel_wrapper<random_access_iterator<int*>>(random_access_iterator<int*>(buffer_ + size_));
  }
  constexpr sentinel_wrapper<contiguous_iterator<const int*>> end() const {
    return sentinel_wrapper<contiguous_iterator<const int*>>(contiguous_iterator<const int*>(buffer_ + size_));
  }
};

constexpr bool test() {
  int buffer[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  {
    // simple-view: const and non-const have the same iterator/sentinel type
    using View = std::ranges::enumerate_view<SimpleNonCommon>;
    static_assert(!std::ranges::common_range<View>);
    static_assert(simple_view<View>);

    View ev{SimpleNonCommon(buffer)};

    assert(ev.begin() != ev.end());
    assert(ev.begin() + 1 != ev.end());
    assert(ev.begin() + 2 != ev.end());
    assert(ev.begin() + 3 != ev.end());
    assert(ev.begin() + 10 == ev.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
