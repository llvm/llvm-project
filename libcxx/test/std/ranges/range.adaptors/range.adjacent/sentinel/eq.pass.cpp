//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

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

template <std::size_t N>
constexpr bool test() {
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  {
    // simple-view: const and non-const have the same iterator/sentinel type
    using View = std::ranges::adjacent_view<SimpleNonCommon, N>;
    static_assert(!std::ranges::common_range<View>);
    static_assert(simple_view<View>);

    View v{SimpleNonCommon(buffer)};

    assert(v.begin() != v.end());
    assert(v.begin() + 1 != v.end());
    assert(v.begin() + 2 != v.end());
    assert(v.begin() + 3 != v.end());
    assert(v.begin() + (10 - N) == v.end());
  }

  {
    // !simple-view: const and non-const have different iterator/sentinel types
    using View = std::ranges::adjacent_view<NonSimpleNonCommon, N>;
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

    View v{NonSimpleNonCommon(buffer)};

    assert(v.begin() != v.end());
    assert(v.begin() + (10 - N) == v.end());

    assert(v.begin() != std::as_const(v).end());
    assert(v.begin() + (10 - N) == std::as_const(v).end());
    // the above works because
    static_assert(std::convertible_to<Iter, ConstIter>);

    assert(std::as_const(v).begin() != std::as_const(v).end());
    assert(std::as_const(v).begin() + (10 - N) == std::as_const(v).end());
  }

  {
    // underlying const/non-const sentinel can be compared with both const/non-const iterator
    using View = std::ranges::adjacent_view<ComparableView, N>;
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

    View v{ComparableView(buffer)};

    assert(v.begin() != v.end());
    assert(v.begin() + (10 - N) == v.end());

    static_assert(!std::convertible_to<Iter, ConstIter>);

    assert(v.begin() != std::as_const(v).end());
    assert(v.begin() + (10 - N) == std::as_const(v).end());

    assert(std::as_const(v).begin() != v.end());
    assert(std::as_const(v).begin() + (10 - N) == v.end());

    assert(std::as_const(v).begin() != std::as_const(v).end());
    assert(std::as_const(v).begin() + (10 - N) == std::as_const(v).end());
  }

  {
    // underlying const/non-const sentinel cannot be compared with non-const/const iterator

    using View = std::ranges::adjacent_view<ConstIncompatibleView, N>;
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
    static_assert(!weakly_equality_comparable_with<Iter, ConstSentinel>);
    static_assert(weakly_equality_comparable_with<ConstIter, ConstSentinel>);

    View v{ConstIncompatibleView{buffer}};

    assert(v.begin() != v.end());
    assert(v.begin() + (10 - N) == v.end());

    assert(std::as_const(v).begin() != std::as_const(v).end());
    assert(std::as_const(v).begin() + (10 - N) == std::as_const(v).end());
  }

  return true;
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
