//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// operator-(x, y)
// operator-(x, default_sentinel )
// operator-(default_sentinel , x)
// operator-(x, n)
// operator+(x, n)
// operator+=(x, n)
// operator-=(x, n)

#include <array>
#include <concepts>
#include <functional>
#include <list>
#include <ranges>
#include <span>
#include <vector>

#include "../../range_adaptor_types.h"

template <class T, class U>
concept canPlusEqual = requires(T& t, U& u) { t += u; };

template <class T, class U>
concept canMinusEqual = requires(T& t, U& u) { t -= u; };

template <class Iter>
struct NotSizedSentinelForIter {
  using iterator_category = std::forward_iterator_tag;
  using value_type        = std::iterator_traits<Iter>::value_type;
  using difference_type   = std::iterator_traits<Iter>::difference_type;

  Iter ptr;

  NotSizedSentinelForIter() = default;
  NotSizedSentinelForIter(const Iter& ptr) : ptr(ptr) {}
  NotSizedSentinelForIter(const NotSizedSentinelForIter& other) : ptr(other.ptr) {}

  value_type& operator*() const { return *ptr; }

  NotSizedSentinelForIter& operator++() { return *++ptr; }
  NotSizedSentinelForIter operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  friend bool operator==(const NotSizedSentinelForIter& a, const NotSizedSentinelForIter& b) = default;
};

template <class T>
struct NotSizedViewWithSizedSentinel;

template <class Iter>
struct SizedSentinelForIter {
  using iterator_category = std::forward_iterator_tag;
  using value_type        = std::iterator_traits<Iter>::value_type;
  using difference_type   = std::iterator_traits<Iter>::difference_type;

  Iter ptr;
  Iter e;

  constexpr SizedSentinelForIter() = default;
  constexpr SizedSentinelForIter(const Iter& ptr, const Iter& e = nullptr) : ptr(ptr), e(e) {}
  constexpr SizedSentinelForIter(const SizedSentinelForIter& other) : ptr(other.ptr) {}

  constexpr value_type& operator*() const { return *ptr; }

  constexpr SizedSentinelForIter& operator++() {
    ++ptr;
    return *this;
  }
  constexpr SizedSentinelForIter operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  friend constexpr decltype(auto) operator-(const SizedSentinelForIter& a, const SizedSentinelForIter& b) {
    return a.ptr - b.ptr;
  }

  friend constexpr bool operator==(const SizedSentinelForIter& a, const SizedSentinelForIter& b) = default;
};

template <class T>
struct SizedViewWithoutSizedSentinel : std::ranges::view_base {
  T* b_;
  std::size_t sz;

  template <std::size_t N>
  constexpr SizedViewWithoutSizedSentinel(T (&b)[N]) : b_(b), sz(N) {}
  constexpr SizedViewWithoutSizedSentinel(T* b, std::size_t N) : b_(b), sz(N) {}

  constexpr NotSizedSentinelForIter<T*> begin() const { return NotSizedSentinelForIter<T*>{b_}; }
  constexpr NotSizedSentinelForIter<T*> end() const { return NotSizedSentinelForIter<T*>{b_ + sz}; }

  constexpr std::size_t size() { return sz; }
};

template <class T>
struct NotSizedViewWithSizedSentinel : std::ranges::view_base {
  T* b_;
  std::size_t sz;

  NotSizedViewWithSizedSentinel() = default;

  template <std::size_t N>
  constexpr NotSizedViewWithSizedSentinel(T (&b)[N]) : b_(b), sz(N) {}
  constexpr NotSizedViewWithSizedSentinel(T* b, std::size_t N) : b_(b), sz(N) {}

  constexpr SizedSentinelForIter<T*> begin() const { return SizedSentinelForIter<T*>{b_, b_ + sz}; }
  constexpr SizedSentinelForIter<T*> end() const { return SizedSentinelForIter<T*>{b_ + sz, b_ + sz}; }
};

static_assert(std::forward_iterator<NotSizedSentinelForIter<int*>>);
static_assert(std::sized_sentinel_for<std::ranges::sentinel_t<NotSizedViewWithSizedSentinel<int>>,
                                      std::ranges::iterator_t<NotSizedViewWithSizedSentinel<int>>>);

template <class... Views>
concept MinusOperatorWellFormedForDefaultSentinel =
    requires(std::ranges::concat_view<Views...> cv) { (cv.begin() - std::default_sentinel_t{}); };

constexpr bool test() {
  int buffer1[5] = {1, 2, 3, 4, 5};
  int buffer2[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  int buffer3[4] = {4, 5, 6, 7};

  SimpleCommonRandomAccessSized a{buffer1};
  SimpleCommonRandomAccessSized b{buffer2};
  SimpleCommonRandomAccessSized c{buffer3};

  {
    // operator+(x, n) and operator+=
    std::ranges::concat_view v(a, b);
    auto it1 = v.begin();

    auto it2 = it1 + 3;
    auto x2  = *it2;
    it2       = it2 + (-2);
    auto x2_1 = *it2;
    assert(x2 == buffer1[3]);
    assert(x2_1 == buffer1[1]);

    auto it3 = 3 + it1;
    auto x3  = *it3;
    it3       = (-2) + it3;
    auto x3_1 = *it3;
    assert(x3 == buffer1[3]);
    assert(x3_1 == buffer1[1]);

    it1 += 3;
    auto x1 = *it1;
    it1 += (-2);
    auto x1_1 = *it1;
    assert(x1 == buffer1[3]);
    assert(x1_1 == buffer1[1]);

    //  reaches the end of one range so it should jump to the beginning of next range
    auto it = v.begin() + 5;
    assert(*it == buffer2[0]);

    using Iter = decltype(it1);
    static_assert(canPlusEqual<Iter, std::intptr_t>);
  }

  {
    // operator+(x, n) and operator+=
    // has empty view
    std::array<int, 0> e{};
    std::ranges::concat_view v(a, e, b);
    auto it1 = v.begin();

    auto it2 = it1 + 3;
    auto x2  = *it2;
    it2       = it2 + (-2);
    auto x2_1 = *it2;
    assert(x2 == buffer1[3]);
    assert(x2_1 == buffer1[1]);

    auto it3 = 3 + it1;
    auto x3  = *it3;
    it3       = (-2) + it3;
    auto x3_1 = *it3;
    assert(x3 == buffer1[3]);
    assert(x3_1 == buffer1[1]);

    it1 += 3;
    auto x1 = *it1;
    it1 += (-2);
    auto x1_1 = *it1;
    assert(x1 == buffer1[3]);
    assert(x1_1 == buffer1[1]);

    // jumps over to next range but next range is empty, so it should go futher
    auto it = v.begin() + 5;
    assert(*it == buffer2[0]);

    using Iter = decltype(it1);
    static_assert(canPlusEqual<Iter, std::intptr_t>);
  }

  {
    // operator+(x, n) and operator+=
    // has more than 2 ranges
    std::array<int, 0> e{};
    std::ranges::concat_view v(a, e, b, c);
    auto it1 = v.begin();

    auto it2 = it1 + 3;
    auto x2  = *it2;
    it2       = it2 + (-2);
    auto x2_1 = *it2;
    assert(x2 == buffer1[3]);
    assert(x2_1 == buffer1[1]);

    auto it3 = 3 + it1;
    auto x3  = *it3;
    it3       = (-2) + it3;
    auto x3_1 = *it3;
    assert(x3 == buffer1[3]);
    assert(x3_1 == buffer1[1]);

    it1 += 3;
    auto x1 = *it1;
    it1 += (-2);
    auto x1_1 = *it1;
    assert(x1 == buffer1[3]);
    assert(x1_1 == buffer1[1]);
    // jumps over to next range but next range is empty, so it should go futher
    auto it = v.begin() + 5;
    assert(*it == buffer2[0]);

    // jumps big enough to skip several number of ranges
    it = v.begin() + 14;
    auto x   = *it;
    it       = it + (-3);
    auto x_1 = *it;
    assert(x == buffer3[0]);
    assert(x_1 == buffer2[6]);

    using Iter = decltype(it1);
    static_assert(canPlusEqual<Iter, std::intptr_t>);
  }

  {
    // operator-(x, n) and operator-=
    std::ranges::concat_view v(a, b);
    auto it1 = v.end();

    auto it2 = it1 - 3;
    auto x2  = *it2;
    it2       = it2 - (-1);
    auto x2_1 = *it2;
    assert(x2 == buffer2[6]);
    assert(x2_1 == buffer2[7]);

    it1 -= 2;
    assert(it1 == it2);
    auto x1 = *it1;
    it1 -= (-1);
    auto x1_1 = *it1;
    assert(x1 == buffer2[7]);
    assert(x1_1 == buffer2[8]);

    // move back to one past the begining,
    // so it should point to the elements of the previous range

    auto it = v.end() - 12;
    auto x  = *it;
    it -= (-1);
    auto x_1 = *it;
    assert(x == buffer1[2]);
    assert(x_1 == buffer1[3]);

    using Iter = decltype(it1);
    static_assert(canMinusEqual<Iter, std::intptr_t>);
  }

  {
    // operator-(x, n) and operator-=
    // has empty view
    std::array<int, 0> e{};
    std::ranges::concat_view v(a, e, b);
    auto it1 = v.end();

    auto it2 = it1 - 3;
    auto x2  = *it2;
    it2       = it2 - (-1);
    auto x2_1 = *it2;
    assert(x2 == buffer2[6]);
    assert(x2_1 == buffer2[7]);

    it1 -= 2;
    assert(it1 == it2);
    auto x1 = *it1;
    it1 -= (-1);
    auto x1_1 = *it1;
    assert(x1 == buffer2[7]);
    assert(x1_1 == buffer2[8]);

    // move back to an empty range
    // so it should skip empty range point to the elements of the previous range

    auto it = v.end() - 12;
    auto x  = *it;
    it -= (-1);
    auto x_1 = *it;
    assert(x == buffer1[2]);
    assert(x_1 == buffer1[3]);

    using Iter = decltype(it1);
    static_assert(canMinusEqual<Iter, std::intptr_t>);
  }

  {
    // operator-(x, n) and operator-=
    // has more than 2 views
    std::array<int, 0> e{};
    std::ranges::concat_view v(a, e, b, c);
    auto it1 = v.end();

    auto it2 = it1 - 3;
    auto x2  = *it2;
    it2 -= (-1);
    auto x2_1 = *it2;
    assert(x2 == buffer3[1]);
    assert(x2_1 == buffer3[2]);

    it1 -= 2;
    assert(it1 == it2);
    auto x1 = *it1;
    it1 -= (-1);
    auto x1_1 = *it1;
    assert(x1 == buffer3[2]);
    assert(x1_1 == buffer3[3]);

    // move back to multiple ranges

    auto it  = v.end() - 16;
    auto x   = *it;
    it       = it - (-1);
    auto x_1 = *it;
    it -= (-1);
    auto x_2 = *it;
    assert(x == buffer1[2]);
    assert(x_1 == buffer1[3]);
    assert(x_2 == buffer1[4]);

    using Iter = decltype(it1);
    static_assert(canMinusEqual<Iter, std::intptr_t>);
  }

  {
    // operator-(x, y)
    // x and y are in different ranges
    // underlying ranges are the same
    // x'index < y's index
    std::ranges::concat_view v(a, b);
    assert((v.end() - v.begin()) == 14);

    // x'index < y's index
    auto it1 = v.begin() + 2;
    auto it2 = v.end() - 1;
    assert((it1 - it2) == -11);

    // x'index > y's index
    assert((it2 - it1) == 11);

    // x'index == y's index
    it1 = it1 + 11;
    assert((it2 - it1) == 0);
  }

  {
    // opeartor-(x,y)
    // x and y are in different ranges
    // underlying ranges are different types
    std::array<int, 3> arr_a{1, 2, 3};
    std::vector<int> arr_b{4, 5, 6};
    std::ranges::concat_view v(arr_a, arr_b);

    // x'index < y's index
    auto it1 = v.begin() + 1;
    auto it2 = v.end() - 1;
    assert(*it1 == 2);
    assert(*it2 == 6);
    assert((it1 - it2) == -4);

    // x'index > y's index
    assert((it2 - it1) == 4);

    // x'index == y's index
    it1 = it1 + 4;
    assert((it2 - it1) == 0);
  }

  {
    // opeartor-(x,y)
    // x and y are in different ranges
    // underlying ranges are different types, and there are empty ranges in the middle
    std::array<int, 0> e_1{};
    std::array<int, 0> e_2{};
    std::vector<int> arr_b{4, 5, 6};
    std::ranges::concat_view v(a, e_1, e_2, arr_b);

    // x'index < y's index
    auto it1 = v.begin() + 1;
    auto it2 = v.end() - 1;
    assert(*it1 == 2);
    assert(*it2 == 6);
    assert((it1 - it2) == -6);

    // x'index > y's index
    assert((it2 - it1) == 6);

    // x'index == y's index
    it1 = it1 + 6;
    assert((it2 - it1) == 0);
  }

  {
    // operator-(default_sentinel , x)
    std::array<int, 2> array1{0, 1};
    std::array<int, 2> array2{2, 3};
    std::array<int, 2> array3{4, 5};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2), std::views::all(array3));
    auto it1 = view.begin();
    auto res = std::default_sentinel_t{} - it1;
    assert(res == 6);
  }

  {
    // operator-(default_sentinel , x) with empty ranges
    std::array<int, 2> array1{0, 1};
    std::array<int, 0> array2;
    std::array<int, 2> array3{2, 3};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2), std::views::all(array3));
    auto it1 = view.begin();
    auto res = std::default_sentinel_t{} - it1;
    assert(res == 4);
  }

  {
    // operator-(default_sentinel , x) with different types
    std::array<int, 2> array1{0, 1};
    std::vector<int> array2{2, 3};
    std::array<int, 0> array3{};
    std::ranges::concat_view view(a, std::views::all(array1), std::views::all(array2), std::views::all(array3));
    auto it1 = view.begin();
    auto res = std::default_sentinel_t{} - it1;
    assert(res == 9);
  }

  {
    // operator-(x, default_sentinel)
    std::array<int, 2> array1{0, 1};
    std::array<int, 2> array2{2, 3};
    std::array<int, 2> array3{4, 5};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2), std::views::all(array3));
    auto it1 = view.begin();
    auto res = it1 - std::default_sentinel_t{};
    assert(res == -6);
  }

  {
    // operator-(x, default_sentinel) with empty ranges
    std::array<int, 2> array1{0, 1};
    std::array<int, 0> array2{};
    std::array<int, 2> array3{4, 5};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2), std::views::all(array3));
    auto it1 = view.begin();
    auto res = it1 - std::default_sentinel_t{};
    assert(res == -4);
  }

  {
    // operator-(x, default_sentinel) with different types
    std::array<int, 2> array1{0, 1};
    std::array<int, 0> array2{};
    std::vector<int> array3{4, 5};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2), std::views::all(array3));
    auto it1 = view.begin();
    auto res = it1 - std::default_sentinel_t{};
    assert(res == -4);
  }

  {
    // operator-(x, default_sentinel)
    // testing constraints

    // sized_sentinel_for fails
    static_assert(!MinusOperatorWellFormedForDefaultSentinel<SizedViewWithoutSizedSentinel<int>>);
    static_assert(!MinusOperatorWellFormedForDefaultSentinel<SizedViewWithoutSizedSentinel<int>,
                                                             NonSimpleCommonRandomAccessSized>);
    static_assert(
        MinusOperatorWellFormedForDefaultSentinel<SimpleCommonRandomAccessSized, NonSimpleCommonRandomAccessSized>);

    // let Fs be the pack containing all views but the first one
    // sized_sentinel_for succeeds and sized_range<Fs...> succeeds
    // first range does not have size() but satisfies sized_sentinel_for
    int arr_a[3] = {0, 1, 2};
    int arr_b[3] = {4, 5, 6};
    static_assert(
        MinusOperatorWellFormedForDefaultSentinel<NotSizedViewWithSizedSentinel<int>, SimpleCommonRandomAccessSized>);
    std::ranges::concat_view cv{NotSizedViewWithSizedSentinel<int>{arr_a}, SimpleCommonRandomAccessSized{arr_b}};
    auto it = cv.begin();
    it++;
    assert((it - std::default_sentinel_t{}) == -5);
    assert((std::default_sentinel_t{} - it) == 5);

    // sized_sentinel_for succeeds but sized_range<Fs...> fails
    static_assert(!MinusOperatorWellFormedForDefaultSentinel<NotSizedViewWithSizedSentinel<int>, InputCommonView>);
    static_assert(!MinusOperatorWellFormedForDefaultSentinel<SimpleCommonRandomAccessSized, InputCommonView>);
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
    // One of the ranges is not random access
    std::ranges::concat_view v(a, b, InputCommonView{buffer1});
    using Iter = decltype(v.begin());
    static_assert(!std::invocable<std::minus<>, Iter, Iter>);
  }

  {
    // random access check
    std::array<int, 4> a1{1, 2, 3, 4};
    std::array<int, 2> b1{5, 6};
    std::span<const int> s1{a1};
    std::span<const int> s2{b1};

    // All random-access & all non-last are common => random access iterator
    {
      auto v      = std::views::concat(s1, s2); // both spans are RA & common; non-last (s1) is common
      using Iter  = decltype(v.begin());
      using CIter = decltype(std::as_const(v).begin());
      static_assert(std::random_access_iterator<Iter>);
      static_assert(std::random_access_iterator<CIter>);
    }

    // Others are common and last is  be non-common => still random access
    {
      auto last_non_common = std::views::counted(a1.data(), static_cast<std::ptrdiff_t>(a1.size()));
      auto v               = std::views::concat(s2, last_non_common); // s2 is common; last is allowed to be non-common
      using Iter           = decltype(v.begin());
      using CIter          = decltype(std::as_const(v).begin());
      static_assert(std::random_access_iterator<Iter>);
      static_assert(std::random_access_iterator<CIter>);
    }

    // a non-last range is non-common => NOT random access
    {
      int buffer[3] = {1, 2, 3};
      auto v        = std::views::concat(SimpleNonCommon{buffer}, s2);
      using Iter    = decltype(v.begin());
      using CIter   = decltype(std::as_const(v).begin());
      static_assert(!std::random_access_iterator<Iter>);
      static_assert(!std::random_access_iterator<CIter>);
    }

    // one underlying range is not random access => NOT random access
    {
      std::list<int> ls{1, 2, 3};
      auto v      = std::views::concat(ls, s2);
      using Iter  = decltype(v.begin());
      using CIter = decltype(std::as_const(v).begin());
      static_assert(!std::random_access_iterator<Iter>);
      static_assert(!std::random_access_iterator<CIter>);
    }
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
