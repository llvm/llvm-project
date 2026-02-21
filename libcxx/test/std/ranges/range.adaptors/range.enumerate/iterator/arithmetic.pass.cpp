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

// class enumerate_view::iterator

// constexpr iterator& operator+=(difference_type x)
//   requires random_access_range<Base>;
// constexpr iterator& operator-=(difference_type x)
//   requires random_access_range<Base>;

//    friend constexpr iterator operator+(const iterator& x, difference_type y)
//      requires random_access_range<Base>;
//    friend constexpr iterator operator+(difference_type x, const iterator& y)
//      requires random_access_range<Base>;
//    friend constexpr iterator operator-(const iterator& x, difference_type y)
//      requires random_access_range<Base>;
//    friend constexpr difference_type operator-(const iterator& x, const iterator& y) noexcept;

#include <concepts>
#include <ranges>

#include "test_iterators.h"

// Range types.

using CommonRandomAccessRange = std::ranges::subrange<int*>;
static_assert(std::ranges::common_range<CommonRandomAccessRange>);
static_assert(std::ranges::random_access_range<CommonRandomAccessRange>);
// clang-format off
static_assert(std::sized_sentinel_for<std::ranges::iterator_t<CommonRandomAccessRange>, 
                                      std::ranges::iterator_t<CommonRandomAccessRange>>);
// clang-format on

using NonCommonRandomAccessRange = std::ranges::subrange<std::counted_iterator<int*>, std::default_sentinel_t>;
static_assert(!std::ranges::common_range<NonCommonRandomAccessRange>);
static_assert(std::ranges::random_access_range<NonCommonRandomAccessRange>);
// clang-format off
static_assert(std::sized_sentinel_for<std::ranges::iterator_t<NonCommonRandomAccessRange>, 
                                      std::ranges::iterator_t<NonCommonRandomAccessRange>>);
// clang-format on

using NonRandomAccessRange = std::ranges::subrange<bidirectional_iterator<int*>>;
static_assert(!std::ranges::random_access_range<NonRandomAccessRange>);
static_assert(std::ranges::bidirectional_range<NonRandomAccessRange>);
// clang-format off
static_assert(!std::sized_sentinel_for<std::ranges::iterator_t<NonRandomAccessRange>, 
                                       std::ranges::iterator_t<NonRandomAccessRange>>);
// clang-format on

template <class BaseRange>
using EnumerateIter = std::ranges::iterator_t<std::ranges::enumerate_view<BaseRange>>;

template <class BaseRange>
using EnumerateSent = std::ranges::sentinel_t<std::ranges::enumerate_view<BaseRange>>;

// Concepts.

template <typename T, typename U>
concept HasPlusEqual = requires(T t, U u) { t += u; };

template <typename T, typename U>
concept HasMinusEqual = requires(T t, U u) { t -= u; };

template <typename T, typename U>
concept HasPlus = requires(T t, U u) { t + u; };

// template <typename T, typename U>
// concept HasMinus = std::input_or_output_iterator<T> && (!std::sentinel_for<U, T> || std::same_as<T, U>) &&
//                    requires(T t, U u) { t - u; };

template <typename T, typename U>
concept HasMinus = std::input_or_output_iterator<T> && requires(T t, U u) {
  // If T and U are both interators, the result is a difference_type otherwise it is an interator.
  { t - u } -> std::same_as<std::conditional_t<std::same_as<T, U>, std::iter_difference_t<T>, T>>;
};

// SFINAE.

static_assert(HasPlusEqual<EnumerateIter<CommonRandomAccessRange>, int>);
static_assert(HasMinusEqual<EnumerateIter<CommonRandomAccessRange>, int>);
static_assert(HasPlus<EnumerateIter<CommonRandomAccessRange>, int>);
static_assert(HasPlus<int, EnumerateIter<CommonRandomAccessRange>>);
static_assert(HasMinus<EnumerateIter<CommonRandomAccessRange>, int>);
static_assert(HasMinus<EnumerateIter<CommonRandomAccessRange>, EnumerateSent<CommonRandomAccessRange>>);

static_assert(HasPlusEqual<EnumerateIter<NonCommonRandomAccessRange>, int>);
static_assert(HasMinusEqual<EnumerateIter<NonCommonRandomAccessRange>, int>);
static_assert(HasPlus<EnumerateIter<NonCommonRandomAccessRange>, int>);
static_assert(HasPlus<int, EnumerateIter<NonCommonRandomAccessRange>>);
static_assert(HasMinus<EnumerateIter<NonCommonRandomAccessRange>, int>);
static_assert(!HasMinus<EnumerateIter<NonCommonRandomAccessRange>, EnumerateSent<NonCommonRandomAccessRange>>);

static_assert(!HasPlusEqual<EnumerateIter<NonRandomAccessRange>, int>);
static_assert(!HasMinusEqual<EnumerateIter<NonRandomAccessRange>, int>);
static_assert(!HasPlus<EnumerateIter<NonRandomAccessRange>, int>);
static_assert(!HasPlus<int, EnumerateIter<NonRandomAccessRange>>);
static_assert(!HasMinus<EnumerateIter<NonRandomAccessRange>, int>);
static_assert(!HasMinus<EnumerateIter<NonRandomAccessRange>, EnumerateSent<NonRandomAccessRange>>);

constexpr void test_with_common_range() {
  int arr[] = {94, 1, 2, 82};
  CommonRandomAccessRange range{arr, arr + 4};

  auto ev = range | std::views::enumerate;

  using DifferenceT = std::ranges::range_difference_t<decltype(ev)>;
  using IteratorT   = std::ranges::iterator_t<decltype(ev)>;

  // constexpr iterator& operator+=(difference_type x)
  {
    auto it = ev.begin();
    constexpr DifferenceT diff{3};
    std::same_as<IteratorT&> decltype(auto) result = (it += diff);

    assert(result.base() == &arr[3]);
    assert(*it.base() == 82);
  }

  // constexpr iterator& operator-=(difference_type x)
  {
    auto it = ev.end();
    constexpr DifferenceT diff{4};
    std::same_as<IteratorT&> decltype(auto) result = (it -= diff);

    assert(result.base() == &arr[0]);
    assert(*it.base() == 94);
  }

  //    friend constexpr iterator operator+(const iterator& x, difference_type y)
  {
    auto it = ev.begin();
    constexpr DifferenceT diff{3};
    std::same_as<IteratorT> decltype(auto) result = (it + diff);

    assert(result.base() == &arr[3]);
    assert(*result.base() == 82);
  }

  //    friend constexpr iterator operator+(difference_type x, const iterator& y)
  {
    auto it = ev.begin();
    constexpr DifferenceT diff{3};
    std::same_as<IteratorT> decltype(auto) result = (diff + it);

    assert(result.base() == &arr[3]);
    assert(*result.base() == 82);
  }

  //    friend constexpr iterator operator-(const iterator& x, difference_type y)
  {
    auto it = ev.end();
    constexpr DifferenceT diff{4};
    std::same_as<IteratorT> decltype(auto) result = (it - diff);

    assert(result.base() == &arr[0]);
    assert(*result.base() == 94);
  }

  //    friend constexpr difference_type operator-(const iterator& x, const iterator& y) noexcept;
  {
    auto it1 = ev.begin();
    auto it2 = ev.end();

    std::same_as<DifferenceT> decltype(auto) result = (it2 - it1);

    assert(result == DifferenceT{4});
    assert(((it2 - DifferenceT{1}) - (it1 + DifferenceT{1})) == DifferenceT{2});

    static_assert(noexcept(it1 - it2));
  }
}

constexpr void test_with_noncommon_range() {
  int arr[]   = {94, 1, 2, 82};
  auto baseIt = std::counted_iterator{arr, 4};
  auto range  = NonCommonRandomAccessRange{baseIt, std::default_sentinel};

  auto ev = range | std::views::enumerate;

  using DifferenceT = std::ranges::range_difference_t<decltype(ev)>;
  using IteratorT   = std::ranges::iterator_t<decltype(ev)>;

  // constexpr iterator& operator+=(difference_type x)
  {
    auto it = ev.begin();
    constexpr DifferenceT diff{3};
    std::same_as<IteratorT&> decltype(auto) result = (it += diff);

    assert(result.base().base() == &arr[3]);
    assert(*it.base() == 82);
  }

  // constexpr iterator& operator-=(difference_type x)
  {
    constexpr DifferenceT diff{4};
    auto it                                        = ev.begin() + diff;
    std::same_as<IteratorT&> decltype(auto) result = (it -= diff);

    assert(result.base().base() == &arr[0]);
    assert(*it.base() == 94);
  }

  //    friend constexpr iterator operator+(const iterator& x, difference_type y)
  {
    auto it = ev.begin();
    constexpr DifferenceT diff{3};
    std::same_as<IteratorT> decltype(auto) result = (it + diff);

    assert(result.base().base() == &arr[3]);
    assert(*result.base() == 82);
  }

  //    friend constexpr iterator operator+(difference_type x, const iterator& y)
  {
    auto it = ev.begin();
    constexpr DifferenceT diff{3};
    std::same_as<IteratorT> decltype(auto) result = (diff + it);

    assert(result.base().base() == &arr[3]);
    assert(*result.base() == 82);
  }

  //    friend constexpr iterator operator-(const iterator& x, difference_type y)
  {
    constexpr DifferenceT diff{4};
    auto it                                       = ev.begin() + diff;
    std::same_as<IteratorT> decltype(auto) result = (it - diff);

    assert(result.base().base() == &arr[0]);
    assert(*result.base() == 94);
  }

  //    friend constexpr difference_type operator-(const iterator& x, const iterator& y) noexcept;
  // Not applicable, see SFINAE.
}

constexpr bool test() {
  test_with_common_range();
  test_with_noncommon_range();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
