//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template <bool OtherConst>
// requires(sized_sentinel_for<sentinel_t<maybe-const<Const, Views>>,
//                             iterator_t<maybe-const<OtherConst, Views>>>&&...)
// friend constexpr common_type_t<range_difference_t<maybe-const<OtherConst, Views>>...>
//   operator-(const iterator<OtherConst>&, const sentinel&)
//
// template <bool OtherConst>
// requires(sized_sentinel_for<sentinel_t<maybe-const<Const, Views>>,
//                             iterator_t<maybe-const<OtherConst, Views>>>&&...)
// friend constexpr common_type_t<range_difference_t<maybe-const<OtherConst, Views>>...>
//   operator-(const sentinel&, const iterator<OtherConst>&)

#include <cassert>
#include <concepts>
#include <functional>
#include <ranges>
#include <tuple>

#include "../types.h"

template <class Base = int*>
struct convertible_forward_sized_iterator {
  Base it_ = nullptr;

  using iterator_category = std::forward_iterator_tag;
  using value_type = int;
  using difference_type = intptr_t;

  convertible_forward_sized_iterator() = default;
  constexpr convertible_forward_sized_iterator(Base it) : it_(it) {}

  template <std::convertible_to<Base> U>
  constexpr convertible_forward_sized_iterator(const convertible_forward_sized_iterator<U>& it) : it_(it.it_) {}

  constexpr decltype(*Base{}) operator*() const { return *it_; }

  constexpr convertible_forward_sized_iterator& operator++() {
    ++it_;
    return *this;
  }
  constexpr convertible_forward_sized_iterator operator++(int) { return forward_sized_iterator(it_++); }

  friend constexpr bool operator==(const convertible_forward_sized_iterator&,
                                   const convertible_forward_sized_iterator&) = default;

  friend constexpr difference_type operator-(const convertible_forward_sized_iterator& x,
                                             const convertible_forward_sized_iterator& y) {
    return x.it_ - y.it_;
  }
};
static_assert(std::forward_iterator<convertible_forward_sized_iterator<>>);

template <class Base>
struct convertible_sized_sentinel {
  Base base_;
  explicit convertible_sized_sentinel() = default;
  constexpr convertible_sized_sentinel(const Base& it) : base_(it) {}

  template <std::convertible_to<Base> U>
  constexpr convertible_sized_sentinel(const convertible_sized_sentinel<U>& other) : base_(other.base_) {}

  template <class U>
    requires(std::convertible_to<Base, U> || std::convertible_to<U, Base>)
  friend constexpr bool operator==(const convertible_sized_sentinel& s, const U& base) {
    return s.base_ == base;
  }
  template <class U>
    requires(std::convertible_to<Base, U> || std::convertible_to<U, Base>)
  friend constexpr auto operator-(const convertible_sized_sentinel& s, const U& i) {
    return s.base_ - i;
  }

  template <class U>
    requires(std::convertible_to<Base, U> || std::convertible_to<U, Base>)
  friend constexpr auto operator-(const U& i, const convertible_sized_sentinel& s) {
    return i - s.base_;
  }
};
static_assert(std::sized_sentinel_for<convertible_sized_sentinel<convertible_forward_sized_iterator<>>,
                                      convertible_forward_sized_iterator<>>);
static_assert(std::sized_sentinel_for<convertible_sized_sentinel<convertible_forward_sized_iterator<const int*>>,
                                      convertible_forward_sized_iterator<int*>>);
static_assert(std::sized_sentinel_for<convertible_sized_sentinel<convertible_forward_sized_iterator<int*>>,
                                      convertible_forward_sized_iterator<const int*>>);

struct ConstCompatibleForwardSized : IntBufferView {
  using IntBufferView::IntBufferView;

  using iterator = convertible_forward_sized_iterator<int*>;
  using const_iterator = convertible_forward_sized_iterator<const int*>;

  constexpr iterator begin() { return {buffer_}; }
  constexpr const_iterator begin() const { return {buffer_}; }
  constexpr convertible_sized_sentinel<iterator> end() { return iterator{buffer_ + size_}; }
  constexpr convertible_sized_sentinel<const_iterator> end() const { return const_iterator{buffer_ + size_}; }
};

// clang-format off
template <class T, class U>
concept HasMinus = std::invocable<std::minus<>,const T&, const U&>;

template <class T>
concept SentinelHasMinus = HasMinus<std::ranges::sentinel_t<T>, std::ranges::iterator_t<T>>;
// clang-format on

constexpr bool test() {
  int buffer1[5] = {1, 2, 3, 4, 5};

  {
    // simple-view
    std::ranges::zip_view v{ForwardSizedNonCommon(buffer1)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    LIBCPP_STATIC_ASSERT(std::ranges::__simple_view<decltype(v)>);

    auto it = v.begin();
    auto st = v.end();
    assert(st - it == 5);
    assert(st - std::ranges::next(it, 1) == 4);

    assert(it - st == -5);
    assert(std::ranges::next(it, 1) - st == -4);
    static_assert(SentinelHasMinus<decltype(v)>);
  }

  {
    // shortest range
    std::ranges::zip_view v(std::views::iota(0, 3), ForwardSizedNonCommon(buffer1));
    static_assert(!std::ranges::common_range<decltype(v)>);
    auto it = v.begin();
    auto st = v.end();
    assert(st - it == 3);
    assert(st - std::ranges::next(it, 1) == 2);

    assert(it - st == -3);
    assert(std::ranges::next(it, 1) - st == -2);
    static_assert(SentinelHasMinus<decltype(v)>);
  }

  {
    // underlying sentinel does not model sized_sentinel_for
    std::ranges::zip_view v(std::views::iota(0), SizedRandomAccessView(buffer1));
    static_assert(!std::ranges::common_range<decltype(v)>);
    static_assert(!SentinelHasMinus<decltype(v)>);
  }

  {
    // const imcompatible:
    // underlying const sentinels cannot substract underlying iterators
    // underlying sentinels cannot substract underlying const iterators
    std::ranges::zip_view v(NonSimpleForwardSizedNonCommon{buffer1});
    static_assert(!std::ranges::common_range<decltype(v)>);
    LIBCPP_STATIC_ASSERT(!std::ranges::__simple_view<decltype(v)>);

    using Iter = std::ranges::iterator_t<decltype(v)>;
    using ConstIter = std::ranges::iterator_t<const decltype(v)>;
    static_assert(!std::is_same_v<Iter, ConstIter>);
    using Sentinel = std::ranges::sentinel_t<decltype(v)>;
    using ConstSentinel = std::ranges::sentinel_t<const decltype(v)>;
    static_assert(!std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(HasMinus<Iter, Sentinel>);
    static_assert(HasMinus<Sentinel, Iter>);
    static_assert(HasMinus<ConstIter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, ConstIter>);
    auto it = v.begin();
    auto const_it = std::as_const(v).begin();
    auto st = v.end();
    auto const_st = std::as_const(v).end();
    assert(it - st == -5);
    assert(st - it == 5);
    assert(const_it - const_st == -5);
    assert(const_st - const_it == 5);

    static_assert(!HasMinus<Iter, ConstSentinel>);
    static_assert(!HasMinus<ConstSentinel, Iter>);
    static_assert(!HasMinus<ConstIter, Sentinel>);
    static_assert(!HasMinus<Sentinel, ConstIter>);
  }

  {
    // const compatible allow non-const to const conversion
    std::ranges::zip_view v(ConstCompatibleForwardSized{buffer1});
    static_assert(!std::ranges::common_range<decltype(v)>);
    LIBCPP_STATIC_ASSERT(!std::ranges::__simple_view<decltype(v)>);

    using Iter = std::ranges::iterator_t<decltype(v)>;
    using ConstIter = std::ranges::iterator_t<const decltype(v)>;
    static_assert(!std::is_same_v<Iter, ConstIter>);
    using Sentinel = std::ranges::sentinel_t<decltype(v)>;
    using ConstSentinel = std::ranges::sentinel_t<const decltype(v)>;
    static_assert(!std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(HasMinus<Iter, Sentinel>);
    static_assert(HasMinus<Sentinel, Iter>);
    static_assert(HasMinus<ConstIter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, ConstIter>);
    static_assert(HasMinus<Iter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, Iter>);
    static_assert(HasMinus<ConstIter, Sentinel>);
    static_assert(HasMinus<Sentinel, ConstIter>);

    auto it = v.begin();
    auto const_it = std::as_const(v).begin();
    auto st = v.end();
    auto const_st = std::as_const(v).end();

    assert(it - st == -5);
    assert(st - it == 5);
    assert(const_it - const_st == -5);
    assert(const_st - const_it == 5);
    assert(it - const_st == -5);
    assert(const_st - it == 5);
    assert(const_it - st == -5);
    assert(st - const_it == 5);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
