//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// operator==(x,y)
// operator==(x, sentinel)
// operator<(x, y)
// operator<=(x, y)
// operator>=(x, y)
// operator>(x, y)
// operator<=>(x, y)

#include <array>
#include <cassert>
#include <ranges>
#include <vector>

#include "../../range_adaptor_types.h"

template <class Iter>
concept canCompareEqual = requires(Iter a, Iter b) { a == b; };

template <class Iter>
concept canCompareEqualWithDefaultSentinel = requires(Iter a, std::default_sentinel_t b) { a == b; };

template <class Iter>
concept canCompareLessthan = requires(Iter a, Iter b) { a < b; };

template <class Iter>
concept canCompareLessthanOrEqual = requires(Iter a, Iter b) { a <= b; };

template <class Iter>
concept canCompareGreaterThan = requires(Iter a, Iter b) { a > b; };

template <class Iter>
concept canCompareGreaterThanOrEqual = requires(Iter a, Iter b) { a >= b; };

template <class Iter>
concept canCompareThreeWay = requires(Iter a, Iter b) { a <=> b; };

constexpr bool test() {
  {
    // test with one view
    std::array<int, 5> array{0, 1, 2, 3, 4};
    std::ranges::concat_view view((array));
    assert(!(view.begin() == view.end()));
    assert(view.begin() != view.end());
    decltype(auto) it1                       = view.begin();
    decltype(auto) it2                       = view.begin();
    std::same_as<bool> decltype(auto) result = (it1 == it2);
    assert(result);

    ++it1;
    assert(!(it1 == it2));
    assert(!(it2 == it1));
  }

  {
    // test with more than one view
    std::array<int, 3> array1{0, 1, 2};
    std::vector<int> array2{0, 1, 2};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    decltype(auto) it1                       = view.begin();
    decltype(auto) it2                       = view.begin();
    std::same_as<bool> decltype(auto) result = (it1 == it2);
    assert(result);

    ++it2;
    ++it2;
    assert(!(it1 == it2));
    assert(!(it2 == it1));
    assert(it2 != it1);
    assert(it1 != it2);
    ++it2;
    assert(*it1 == *it2);
    assert(*it2 == *it1);
    assert(!(*it1 != *it2));
    assert(!(*it2 != *it1));
  }

  {
    // test with more than one view and iterators are in different range
    std::array<int, 3> array1{0, 1, 2};
    std::vector<int> array2{4, 5, 6};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    decltype(auto) it1 = view.begin();
    decltype(auto) it2 = view.begin() + 3;

    assert(it1 != it2);
    assert(it2 != it1);
    assert(!(it1 == it2));
    assert(!(it2 == it1));
    assert(*it1 == 0);
    assert(*it2 == 4);
    it1++;
    it2++;
    assert(*it1 == 1);
    assert(*it2 == 5);
  }

  {
    // operator==(x, sentinel)
    std::array<int, 2> array1{1, 2};
    std::vector<int> array2{3, 4};
    std::ranges::concat_view v(std::views::all(array1), std::views::all(array2));

    auto it = v.begin();
    assert(!(it == std::default_sentinel_t{}));
    assert(!(std::default_sentinel_t{} == it));
    assert(it != std::default_sentinel_t{});
    assert(std::default_sentinel_t{} != it);

    it++;
    it++;
    it++;
    it++;
    assert(it == std::default_sentinel_t{});
    assert(std::default_sentinel_t{} == it);
    assert(!(it != std::default_sentinel_t{}));
    assert(!(std::default_sentinel_t{} != it));

    // const-iterator
    const auto& cv = v;
    auto cit       = cv.begin();
    ++cit;
    ++cit;
    ++cit;
    ++cit;
    assert(cit == std::default_sentinel_t{});
    assert(std::default_sentinel_t{} == cit);
    assert(!(cit != std::default_sentinel_t{}));
    assert(!(std::default_sentinel_t{} != cit));
  }

  {
    // operator <, <=, >, >=
    std::array<int, 4> arr_a{1, 2, 3, 4};
    std::vector<int> arr_b{5, 6, 7};
    auto v = std::views::concat(arr_a, arr_b);
    auto i = v.begin();
    auto j = v.begin();
    std::ranges::advance(j, arr_a.size());

    assert(i < j);
    assert(i <= j);
    assert(!(i > j));
    assert(!(i >= j));
    assert((i <=> j) == std::strong_ordering::less);
    assert((i <=> i) == std::strong_ordering::equal);
    assert((j <=> i) == std::strong_ordering::greater);

    auto k = j;
    assert(!(j < k));
    assert(j <= k);
    assert(!(j > k));
    assert(j >= k);
    auto ord2 = (j <=> k);
    assert(ord2 == 0);

    // const-iterator
    const auto& cv = v;
    auto ci        = cv.begin();
    auto cj        = cv.begin();
    std::ranges::advance(cj, arr_a.size());
    assert(ci < cj);
    assert((ci <=> cj) < 0);
  }

  {
    // operator <, <=, >, >=
    // two pointers point to elements in the same range
    std::array<int, 4> arr_a{1, 2, 3, 4};
    std::vector<int> arr_b{5, 6, 7};
    auto v = std::views::concat(arr_a, arr_b);
    auto i = v.begin();
    auto j = v.begin() + 2;

    assert(i < j);
    assert(i <= j);
    assert(!(i > j));
    assert(!(i >= j));
    assert((i <=> j) == std::strong_ordering::less);
    assert((i <=> i) == std::strong_ordering::equal);
    assert((j <=> i) == std::strong_ordering::greater);

    auto k = j;
    assert(!(j < k));
    assert(j <= k);
    assert(!(j > k));
    assert(j >= k);
    auto ord2 = (j <=> k);
    assert(ord2 == 0);

    // const-iterator
    const auto& cv = v;
    auto ci        = cv.begin();
    auto cj        = cv.begin();
    std::ranges::advance(cj, arr_a.size());
    assert(ci < cj);
    assert((ci <=> cj) < 0);
  }

  {
    // operator ==,<, <=, >, >=, <=>
    // should not be invocable on non-random access range
    static_assert(!canCompareEqual<std::ranges::concat_view<ForwardSizedView>>);
    static_assert(!canCompareEqualWithDefaultSentinel<std::ranges::concat_view<ForwardSizedView>>);
    static_assert(!canCompareLessthan<std::ranges::concat_view<ForwardSizedView>>);
    static_assert(!canCompareLessthanOrEqual<std::ranges::concat_view<ForwardSizedView>>);
    static_assert(!canCompareGreaterThan<std::ranges::concat_view<ForwardSizedView>>);
    static_assert(!canCompareGreaterThanOrEqual<std::ranges::concat_view<ForwardSizedView>>);
    static_assert(!canCompareThreeWay<std::ranges::concat_view<ForwardSizedView>>);

    static_assert(!canCompareEqual<std::ranges::concat_view<ForwardSizedView, SizedBidiCommon>>);
    static_assert(!canCompareEqualWithDefaultSentinel<std::ranges::concat_view<ForwardSizedView, SizedBidiCommon>>);
    static_assert(!canCompareLessthan<std::ranges::concat_view<ForwardSizedView, SizedBidiCommon>>);
    static_assert(!canCompareLessthanOrEqual<std::ranges::concat_view<ForwardSizedView, SizedBidiCommon>>);
    static_assert(!canCompareGreaterThan<std::ranges::concat_view<ForwardSizedView, SizedBidiCommon>>);
    static_assert(!canCompareGreaterThanOrEqual<std::ranges::concat_view<ForwardSizedView, SizedBidiCommon>>);
    static_assert(!canCompareThreeWay<std::ranges::concat_view<ForwardSizedView, SizedBidiCommon>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
