//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<range R>
//   constexpr range_difference_t<R> ranges::distance(R&& r);

#include <iterator>
#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "test_macros.h"

template<class It, class Sent>
constexpr void test_ordinary() {
  struct R {
    mutable int a[3] = {1, 2, 3};
    constexpr It begin() const { return It(a); }
    constexpr Sent end() const { return Sent(It(a + 3)); }
  };
  R r;
  assert(std::ranges::distance(r) == 3);
  assert(std::ranges::distance(static_cast<R&&>(r)) == 3);
  assert(std::ranges::distance(static_cast<const R&>(r)) == 3);
  assert(std::ranges::distance(static_cast<const R&&>(r)) == 3);
  ASSERT_SAME_TYPE(decltype(std::ranges::distance(r)), std::ranges::range_difference_t<R>);
}

constexpr bool test() {
  {
    using R = int[3];
    int a[] = {1, 2, 3};
    assert(std::ranges::distance(static_cast<R&>(a)) == 3);
    assert(std::ranges::distance(static_cast<R&&>(a)) == 3);
    assert(std::ranges::distance(static_cast<const R&>(a)) == 3);
    assert(std::ranges::distance(static_cast<const R&&>(a)) == 3);
    ASSERT_SAME_TYPE(decltype(std::ranges::distance(a)), std::ptrdiff_t);
    ASSERT_SAME_TYPE(decltype(std::ranges::distance(a)), std::ranges::range_difference_t<R>);
  }
  {
    // Unsized range, non-copyable iterator type, rvalue-ref-qualified begin()
    using It = cpp20_input_iterator<int*>;
    using Sent = sentinel_wrapper<cpp20_input_iterator<int*>>;
    using R = std::ranges::subrange<It, Sent, std::ranges::subrange_kind::unsized>;

    int a[] = {1, 2, 3};
    auto r = R(It(a), Sent(It(a + 3)));
    assert(std::ranges::distance(r) == 3);
    assert(std::ranges::distance(static_cast<R&&>(r)) == 3);
    static_assert(!std::is_invocable_v<decltype(std::ranges::distance), const R&>);
    static_assert(!std::is_invocable_v<decltype(std::ranges::distance), const R&&>);
  }
  {
    // Sized range (unsized sentinel type), non-copyable iterator type, rvalue-ref-qualified begin()
    using It = cpp20_input_iterator<int*>;
    using Sent = sentinel_wrapper<cpp20_input_iterator<int*>>;
    using R = std::ranges::subrange<It, Sent, std::ranges::subrange_kind::sized>;

    int a[] = {1, 2, 3};
    auto r = R(It(a), Sent(It(a + 3)), 3);
    assert(std::ranges::distance(r) == 3);
    assert(std::ranges::distance(static_cast<R&&>(r)) == 3);
    static_assert(!std::is_invocable_v<decltype(std::ranges::distance), const R&>);
    static_assert(!std::is_invocable_v<decltype(std::ranges::distance), const R&&>);
  }
  {
    // Sized range (sized sentinel type), non-copyable iterator type
    test_ordinary<cpp20_input_iterator<int*>, sized_sentinel<cpp20_input_iterator<int*>>>();
  }
  test_ordinary<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_ordinary<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_ordinary<output_iterator<int*>, sentinel_wrapper<output_iterator<int*>>>();
  test_ordinary<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>();
  test_ordinary<bidirectional_iterator<int*>, sentinel_wrapper<bidirectional_iterator<int*>>>();
  test_ordinary<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>();
  test_ordinary<contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>>();
  test_ordinary<int*, sentinel_wrapper<int*>>();

  test_ordinary<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>();
  test_ordinary<cpp20_input_iterator<int*>, sized_sentinel<cpp20_input_iterator<int*>>>();
  test_ordinary<output_iterator<int*>, sized_sentinel<output_iterator<int*>>>();
  test_ordinary<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>();
  test_ordinary<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>();
  test_ordinary<random_access_iterator<int*>, sized_sentinel<random_access_iterator<int*>>>();
  test_ordinary<contiguous_iterator<int*>, sized_sentinel<contiguous_iterator<int*>>>();
  test_ordinary<int*, sized_sentinel<int*>>();
  test_ordinary<int*, int*>();

  // Calling it on a non-range isn't allowed.
  static_assert(!std::is_invocable_v<decltype(std::ranges::distance), int>);
  static_assert(!std::is_invocable_v<decltype(std::ranges::distance), int*>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
