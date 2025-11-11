//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// operator-(x, y)
// operator-(x, sentinel)
// operator-(sentinel, x)
// operator-(x, n)
// operator+(x, n)
// operator+=(x, n)
// operator-=(x, n)
// operator<(x, y)
// operator<=(x, y)
// operator>=(x, y)
// operator>(x, y)
// operator<=>(x, y)

#include <array>
#include <concepts>
#include <functional>
#include <list>
#include <ranges>
#include <span>

#include "../../range_adaptor_types.h"

template <class T, class U>
concept canPlusEqual = requires(T& t, U& u) { t += u; };

template <class T, class U>
concept canMinusEqual = requires(T& t, U& u) { t -= u; };

constexpr bool test() {
  int buffer1[5] = {1, 2, 3, 4, 5};
  int buffer2[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  SimpleCommonRandomAccessSized a{buffer1};
  SimpleCommonRandomAccessSized b{buffer2};

  {
    // operator+(x, n) and operator+=
    std::ranges::concat_view v(a, b);
    auto it1 = v.begin();

    auto it2 = it1 + 3;
    auto x2  = *it2;
    assert(x2 == buffer1[3]);

    auto it3 = 3 + it1;
    auto x3  = *it3;
    assert(x3 == buffer1[3]);

    it1 += 3;
    assert(it1 == it2);
    auto x1 = *it2;
    assert(x1 == buffer1[3]);

    using Iter = decltype(it1);
    static_assert(canPlusEqual<Iter, std::intptr_t>);
  }

  {
    // operator-(x, n) and operator-=
    std::ranges::concat_view v(a, b);
    auto it1 = v.end();

    auto it2 = it1 - 3;
    auto x2  = *it2;
    assert(x2 == buffer2[6]);

    it1 -= 3;
    assert(it1 == it2);
    auto x1 = *it2;
    assert(x1 == buffer2[6]);

    using Iter = decltype(it1);
    static_assert(canMinusEqual<Iter, std::intptr_t>);
  }

  {
    // operator-(x, y)
    std::ranges::concat_view v(a, b);
    assert((v.end() - v.begin()) == 14);

    auto it1 = v.begin() + 2;
    auto it2 = v.end() - 1;
    assert((it1 - it2) == -11);
  }

  {
    // operator-(sentinel, x)
    std::array<int, 2> array1{0, 1};
    std::array<int, 2> array2{2, 3};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    auto it1 = view.begin();
    auto res = std::default_sentinel_t{} - it1;
    assert(res == 4);
  }

  {
    // operator-(x, sentinel)
    std::array<int, 2> array1{0, 1};
    std::array<int, 2> array2{2, 3};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    auto it1 = view.begin();
    auto res = it1 - std::default_sentinel_t{};
    assert(res == -4);
  }

  {
    // One of the ranges is not random access
    std::ranges::concat_view v(a, b, ForwardSizedView{buffer1});
    using Iter = decltype(v.begin());
    static_assert(!std::invocable<std::plus<>, Iter, std::intptr_t>);
    static_assert(!std::invocable<std::plus<>, std::intptr_t, Iter>);
    static_assert(!canPlusEqual<Iter, std::intptr_t>);
    static_assert(!std::invocable<std::minus<>, Iter, std::intptr_t>);
    static_assert(!std::invocable<std::minus<>, Iter, Iter>);
    static_assert(!canMinusEqual<Iter, std::intptr_t>);
  }

  {
    // One of the ranges does not have sized sentinel
    std::ranges::concat_view v(a, b, InputCommonView{buffer1});
    using Iter = decltype(v.begin());
    static_assert(!std::invocable<std::minus<>, Iter, Iter>);
  }

  {
    // random access check
    std::array<int, 4> a1{1,2,3,4};
    std::array<int, 2> b1{5,6};
    std::span<const int> s1{a1};
    std::span<const int> s2{b1};

     // All random-access & all non-last are common => random access iterator
    {
      auto v = std::views::concat(s1, s2); // both spans are RA & common; non-last (s1) is common
      using Iter = decltype(v.begin());
      using CIter = decltype(std::as_const(v).begin());
      static_assert(std::random_access_iterator<Iter>);
      static_assert(std::random_access_iterator<CIter>);
    }

    // Others are common and last is  be non-common => still random access
    {
      auto last_non_common = std::views::counted(a1.data(), static_cast<std::ptrdiff_t>(a1.size()));
      auto v = std::views::concat(s2, last_non_common); // s2 is common; last is allowed to be non-common
      using Iter = decltype(v.begin());
      using CIter = decltype(std::as_const(v).begin());
      static_assert(std::random_access_iterator<Iter>);
      static_assert(std::random_access_iterator<CIter>);
    }

    // a non-last range is non-common => NOT random access
    {
      int buffer[3] = {1, 2, 3};
      auto v = std::views::concat(SimpleNonCommon{buffer}, s2);
      using Iter = decltype(v.begin());
      using CIter = decltype(std::as_const(v).begin());
      static_assert(!std::random_access_iterator<Iter>);
      static_assert(!std::random_access_iterator<CIter>);
    }

    // one underlying range is not random access => NOT random access
    {
      std::list<int> ls{1,2,3};
      auto v = std::views::concat(ls, s2);
      using Iter = decltype(v.begin());
      using CIter = decltype(std::as_const(v).begin());
      static_assert(!std::random_access_iterator<Iter>);
      static_assert(!std::random_access_iterator<CIter>);
    }
  }

  {
    // operator <, <=, >, >=
    std::array<int, 4> arr_a{1, 2, 3, 4};
    std::array<int, 3> arr_b{5, 6, 7};
    std::span<const int> s1{arr_a};
    std::span<const int> s2{arr_b};
    auto v      = std::views::concat(s1, s2);
    using Iter  = decltype(v.begin());
    using CIter = decltype(std::as_const(v).begin());
    auto i      = v.begin();
    auto j      = v.begin();
    std::ranges::advance(j, arr_a.size());

    assert(i < j);
    assert(i <= j);
    assert(!(i > j));
    assert(!(i >= j));
    auto ord1 = (i <=> j);
    assert(ord1 < 0);
    assert((j <=> i) > 0);

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
    // operator+(x, n) and operator+(n, x), where n is negative
    std::array<int, 4> arr_a{1, 2, 3, 4};
    std::array<int, 3> arr_b{5, 6, 7};
    std::span<const int> s1{arr_a};
    std::span<const int> s2{arr_b};
    auto v = std::views::concat(s1, s2);

    auto i = v.begin();
    std::ranges::advance(i, arr_a.size());

    auto j   = i;
    auto j_1 = j + (-1);
    assert(*j_1 == arr_a.back());
    auto j_3 = j + (-3);
    assert(*j_3 == arr_a[1]);

    // n + x (negative)
    auto k = (-1) + j;
    assert(*k == arr_a.back());

    // const-iterator
    auto ci = std::as_const(v).begin();
    std::ranges::advance(ci, arr_a.size());
    auto cj   = ci;
    auto cj_2 = cj + (-2);
    assert(*cj_2 == arr_a[2]);
    auto cjn = (-1) + cj;
    assert(*cjn == arr_a.back());
  }

  {
    // operator-(x, n), where n is negative
    std::array<int, 4> arr_a{1, 2, 3, 4};
    std::array<int, 3> arr_b{5, 6, 7};
    std::span<const int> s1{arr_a};
    std::span<const int> s2{arr_b};
    auto v = std::views::concat(s1, s2);

    auto i = v.begin();

    auto j   = i;
    auto j_1 = j - (-1);
    assert(*j_1 == arr_a[1]);
    auto j_3 = j - (-3);
    assert(*j_3 == arr_a.back());

    // const-iterator
    auto ci   = std::as_const(v).begin();
    auto cj   = ci;
    auto cj_2 = cj - (-2);
    assert(*cj_2 == arr_a[2]);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
