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
#include <concepts>
#include <ranges>
#include <type_traits>
#include <utility>

#include "../../range_adaptor_types.h"
#include "../types.h"
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
  int buffer[] = {0, 1, 2, 3, 85};
  {
    // simple-view: const and non-const have the same iterator/sentinel type
    using View = std::ranges::enumerate_view<SimpleNonCommon>;
    static_assert(!std::ranges::common_range<View>);
    static_assert(simple_view<View>);

    static_assert(weakly_equality_comparable_with<std::ranges::iterator_t<View>, std::ranges::sentinel_t<View>>);

    View ev{SimpleNonCommon(buffer)};

    assert(ev.begin() != ev.end());
    assert(ev.begin() + 1 != ev.end());
    assert(ev.begin() + 2 != ev.end());
    assert(ev.begin() + 3 != ev.end());
    assert(ev.begin() + 5 == ev.end());
  }

  {
    // !simple-view: const and non-const have different iterator/sentinel types
    using View = std::ranges::enumerate_view<NonSimpleNonCommon>;
    static_assert(!std::ranges::common_range<View>);
    static_assert(!simple_view<View>);

    using Iter      = std::ranges::iterator_t<View>;
    using ConstIter = std::ranges::iterator_t<const View>;
    static_assert(!std::is_same_v<Iter, ConstIter>);

    using Sentinel      = std::ranges::sentinel_t<View>;
    using ConstSentinel = std::ranges::sentinel_t<const View>;
    static_assert(!std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(weakly_equality_comparable_with<Iter, Sentinel>);
    static_assert(!weakly_equality_comparable_with<ConstIter, Sentinel>);
    static_assert(weakly_equality_comparable_with<Iter, ConstSentinel>);
    static_assert(weakly_equality_comparable_with<ConstIter, ConstSentinel>);

    View ev{NonSimpleNonCommon(buffer)};

    assert(ev.begin() != ev.end());
    assert(ev.begin() + 5 == ev.end());

    assert(ev.begin() != std::as_const(ev).end());
    assert(ev.begin() + 5 == std::as_const(ev).end());
    // the above works because
    static_assert(std::convertible_to<Iter, ConstIter>);

    assert(std::as_const(ev).begin() != std::as_const(ev).end());
    assert(std::as_const(ev).begin() + 5 == std::as_const(ev).end());
  }

  {
    // underlying const/non-const sentinel can be compared with both const/non-const iterator
    using View = std::ranges::enumerate_view<ComparableView>;
    static_assert(!std::ranges::common_range<View>);
    static_assert(!simple_view<View>);

    using Iter      = std::ranges::iterator_t<View>;
    using ConstIter = std::ranges::iterator_t<const View>;
    static_assert(!std::is_same_v<Iter, ConstIter>);

    using Sentinel      = std::ranges::sentinel_t<View>;
    using ConstSentinel = std::ranges::sentinel_t<const View>;
    static_assert(!std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(weakly_equality_comparable_with<Iter, Sentinel>);
    static_assert(weakly_equality_comparable_with<ConstIter, Sentinel>);
    static_assert(weakly_equality_comparable_with<Iter, ConstSentinel>);
    static_assert(weakly_equality_comparable_with<ConstIter, ConstSentinel>);

    View ev{ComparableView(buffer)};

    assert(ev.begin() != ev.end());
    assert(ev.begin() + 5 == ev.end());

    static_assert(!std::convertible_to<Iter, ConstIter>);

    assert(ev.begin() != std::as_const(ev).end());
    assert(ev.begin() + 5 == std::as_const(ev).end());

    assert(std::as_const(ev).begin() != ev.end());
    assert(std::as_const(ev).begin() + 5 == ev.end());

    assert(std::as_const(ev).begin() != std::as_const(ev).end());
    assert(std::as_const(ev).begin() + 5 == std::as_const(ev).end());
  }

  {
    // underlying const/non-const sentinel cannot be compared with non-const/const iterator
    using View = std::ranges::enumerate_view<ComparableView>;
    static_assert(!std::ranges::common_range<View>);
    static_assert(!simple_view<View>);

    using Iter      = std::ranges::iterator_t<View>;
    using ConstIter = std::ranges::iterator_t<const View>;
    static_assert(!std::is_same_v<Iter, ConstIter>);

    using Sentinel      = std::ranges::sentinel_t<View>;
    using ConstSentinel = std::ranges::sentinel_t<const View>;
    static_assert(!std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(weakly_equality_comparable_with<Iter, Sentinel>);
    static_assert(weakly_equality_comparable_with<ConstIter, Sentinel>);
    static_assert(weakly_equality_comparable_with<Iter, ConstSentinel>);
    static_assert(weakly_equality_comparable_with<ConstIter, ConstSentinel>);

    View ev{ComparableView(buffer)};

    assert(ev.begin() != ev.end());
    assert(ev.begin() + 5 == ev.end());

    assert(std::as_const(ev).begin() != std::as_const(ev).end());
    assert(std::as_const(ev).begin() + 5 == std::as_const(ev).end());
  }

  {
    // sentinel comparison with input move-only iterator must work without copying the iterator
    using InputIterator = cpp20_input_iterator<int*>;
    using Sentinel      = sentinel_wrapper<InputIterator>;
    using View          = MinimalView<InputIterator, Sentinel>;
    static_assert(simple_view<View>);

    View mv{InputIterator(buffer), Sentinel(InputIterator(buffer + 5))};
    std::ranges::enumerate_view ev(std::move(mv));

    auto it = ev.begin();
    assert(it != ev.end());
    assert(ev.end() != it);

    // enumerate_view iterator only has operator+ when the underlying range is random-access.
    // for input_iterator we increment it explicitly.
    std::ranges::advance(it, 5);

    assert(it == ev.end());
    assert(ev.end() == it);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
