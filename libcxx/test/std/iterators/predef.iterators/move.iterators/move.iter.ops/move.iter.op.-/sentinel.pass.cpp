//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// move_iterator

// template<sized_sentinel_for<Iterator> S>
//   friend constexpr iter_difference_t<Iterator>
//     operator-(const move_sentinel<S>& x, const move_iterator& y); // Since C++20
// template<sized_sentinel_for<Iterator> S>
//   friend constexpr iter_difference_t<Iterator>
//     operator-(const move_iterator& x, const move_sentinel<S>& y); // Since C++20

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

// The `operator-` calls the underlying iterator and sentinel's `operator-`.

struct CustomIt {
  using value_type = int;
  using difference_type = int;
  using reference = int&;
  using pointer = int*;
  using iterator_category = std::input_iterator_tag;

  CustomIt() = default;
  TEST_CONSTEXPR_CXX17 explicit CustomIt(int* p) : p_(p) {}
  int& operator*() const;
  CustomIt& operator++();
  CustomIt operator++(int);
  constexpr friend difference_type operator-(const CustomIt& a, const CustomIt& b) { return a.p_ - b.p_; }
  int *p_ = nullptr;
};

template <class It, class Sent = sized_sentinel<It>>
constexpr void test_one() {
  int arr[] = {3, 1, 4};

  const std::move_iterator<It> it_a{It(arr)};
  const std::move_iterator<It> it_b{It(arr + 1)};

  const std::move_sentinel<Sent> sent_a{Sent(It(arr))};
  const std::move_sentinel<Sent> sent_b{Sent(It(arr + 1))};
  const std::move_sentinel<Sent> sent_c{Sent(It(arr + 2))};

  ASSERT_SAME_TYPE(decltype(it_a - sent_a), std::iter_difference_t<It>);
  ASSERT_SAME_TYPE(decltype(sent_a - it_a), std::iter_difference_t<It>);

  // it_a
  assert(it_a - sent_a == 0);
  assert(sent_a - it_a == 0);

  assert(it_a - sent_b == -1);
  assert(sent_b - it_a == 1);

  assert(it_a - sent_c == -2);
  assert(sent_c - it_a == 2);

  // it_b
  assert(it_b - sent_a == 1);
  assert(sent_a - it_b == -1);

  assert(it_b - sent_b == 0);
  assert(sent_b - it_b == 0);

  assert(it_b - sent_c == -1);
  assert(sent_c - it_b == 1);
}

constexpr bool test() {
  test_one<CustomIt>();
  test_one<cpp17_input_iterator<int*>>();
  test_one<forward_iterator<int*>>();
  test_one<bidirectional_iterator<int*>>();
  test_one<random_access_iterator<int*>>();
  test_one<int*>();
  test_one<const int*>();
  test_one<contiguous_iterator<int*>>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
