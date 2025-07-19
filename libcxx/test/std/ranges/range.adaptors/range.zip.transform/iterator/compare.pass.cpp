//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// friend constexpr bool operator==(const iterator& x, const iterator& y)
//   requires equality_comparable<ziperator<Const>>;

// friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;

#include <ranges>

#include <compare>

#include "test_iterators.h"
#include "../types.h"

constexpr void compareOperatorTest(auto&& iter1, auto&& iter2) {
  assert(!(iter1 < iter1));
  assert(iter1 < iter2);
  assert(!(iter2 < iter1));
  assert(iter1 <= iter1);
  assert(iter1 <= iter2);
  assert(!(iter2 <= iter1));
  assert(!(iter1 > iter1));
  assert(!(iter1 > iter2));
  assert(iter2 > iter1);
  assert(iter1 >= iter1);
  assert(!(iter1 >= iter2));
  assert(iter2 >= iter1);
  assert(iter1 == iter1);
  assert(!(iter1 == iter2));
  assert(iter2 == iter2);
  assert(!(iter1 != iter1));
  assert(iter1 != iter2);
  assert(!(iter2 != iter2));
}

constexpr void inequalityOperatorsDoNotExistTest(auto&& iter1, auto&& iter2) {
  using Iter1 = decltype(iter1);
  using Iter2 = decltype(iter2);
  static_assert(!std::is_invocable_v<std::less<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::less_equal<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::greater<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::greater_equal<>, Iter1, Iter2>);
}

constexpr bool test() {
  {
    // Test a new-school iterator with operator<=>; the iterator should also have operator<=>.
    using It       = three_way_contiguous_iterator<int*>;
    using SubRange = std::ranges::subrange<It>;
    static_assert(std::three_way_comparable<It>);

    int a[]    = {1, 2, 3, 4};
    int b[]    = {5, 6, 7, 8, 9};
    auto r     = std::views::zip_transform(MakeTuple{}, SubRange(It(a), It(a + 4)), SubRange(It(b), It(b + 5)));
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;
    using Iter = decltype(iter1);
    static_assert(std::three_way_comparable<Iter>);
    compareOperatorTest(iter1, iter2);

    assert((iter1 <=> iter2) == std::strong_ordering::less);
    assert((iter1 <=> iter1) == std::strong_ordering::equal);
    assert((iter2 <=> iter1) == std::strong_ordering::greater);
  }

  {
    // Test an old-school iterator with no operator<=>; the transform iterator shouldn't have
    // operator<=> either.
    using It       = random_access_iterator<int*>;
    using Subrange = std::ranges::subrange<It>;
    static_assert(!std::three_way_comparable<It>);

    int a[]    = {1, 2, 3, 4};
    int b[]    = {5, 6, 7, 8, 9};
    auto r     = std::views::zip_transform(MakeTuple{}, Subrange(It(a), It(a + 4)), Subrange(It(b), It(b + 5)));
    auto iter1 = r.begin();
    using Iter = decltype(iter1);
#ifndef _LIBCPP_VERSION
    // libc++ hasn't implemented LWG-3692 "zip_transform_view::iterator's operator<=> is overconstrained"
    auto iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);
    static_assert(std::three_way_comparable<Iter>);
    assert((iter1 <=> iter2) == std::strong_ordering::less);
    assert((iter1 <=> iter1) == std::strong_ordering::equal);
    assert((iter2 <=> iter1) == std::strong_ordering::greater);
#endif
  }

  {
    // non random_access_range
    int buffer1[1] = {1};
    int buffer2[2] = {1, 2};

    std::ranges::zip_transform_view v{MakeTuple{}, InputCommonView(buffer1), InputCommonView(buffer2)};
    using View = decltype(v);
    static_assert(!std::ranges::forward_range<View>);
    static_assert(std::ranges::input_range<View>);
    static_assert(std::ranges::common_range<View>);

    auto it1 = v.begin();
    auto it2 = v.end();
    assert(it1 != it2);

    ++it1;
    assert(it1 == it2);

    inequalityOperatorsDoNotExistTest(it1, it2);
  }

  {
    // in this case sentinel is computed by getting each of the underlying sentinel, so only one
    // underlying iterator is comparing equal
    int buffer1[1] = {1};
    int buffer2[2] = {1, 2};
    std::ranges::zip_transform_view v{MakeTuple{}, ForwardSizedView(buffer1), ForwardSizedView(buffer2)};
    using View = decltype(v);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::bidirectional_range<View>);

    auto it1 = v.begin();
    auto it2 = v.end();
    assert(it1 != it2);

    ++it1;
    // it1:  <buffer1 + 1, buffer2 + 1>
    // it2:  <buffer1 + 1, buffer2 + 2>
    assert(it1 == it2);

    inequalityOperatorsDoNotExistTest(it1, it2);
  }

  {
    // underlying iterator does not support ==
    using IterNoEqualView = BasicView<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>;
    int buffer[]          = {1};
    std::ranges::zip_transform_view r(MakeTuple{}, IterNoEqualView{buffer});
    auto it    = r.begin();
    using Iter = decltype(it);
    static_assert(!std::invocable<std::equal_to<>, Iter, Iter>);
    inequalityOperatorsDoNotExistTest(it, it);
  }
  return true;
}

int main(int, char**) {
  test();
  //static_assert(test());

  return 0;
}
