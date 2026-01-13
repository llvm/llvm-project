//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <InputIterator Iter>
//   Iter::difference_type
//   distance(Iter first, Iter last); // constexpr in C++17
//
// template <RandomAccessIterator Iter>
//   Iter::difference_type
//   distance(Iter first, Iter last); // constexpr in C++17

#include <array>
#include <cassert>
#include <deque>
#include <iterator>
#include <vector>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
TEST_CONSTEXPR_CXX17 void check_distance(It first, It last, typename std::iterator_traits<It>::difference_type dist) {
  typedef typename std::iterator_traits<It>::difference_type Difference;
  static_assert(std::is_same<decltype(std::distance(first, last)), Difference>::value, "");
  assert(std::distance(first, last) == dist);
}

#if TEST_STD_VER >= 20
/*TEST_CONSTEXPR_CXX26*/ void test_deque() { // TODO: Mark as TEST_CONSTEXPR_CXX26 once std::deque is constexpr
  using Container = std::deque<std::deque<double>>;
  Container c;
  auto view                    = c | std::views::join;
  Container::difference_type n = 0;
  for (std::size_t i = 0; i < 10; ++i) {
    n += i;
    c.push_back(Container::value_type(i));
  }
  assert(std::distance(view.begin(), view.end()) == n);
}
#endif

TEST_CONSTEXPR_CXX17 bool tests() {
  const char* s = "1234567890";
  check_distance(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 10), 10);
  check_distance(forward_iterator<const char*>(s), forward_iterator<const char*>(s + 10), 10);
  check_distance(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s + 10), 10);
  check_distance(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s + 10), 10);
  check_distance(s, s + 10, 10);

#if TEST_STD_VER >= 20
  {
    using Container = std::vector<std::vector<int>>;
    Container c;
    auto view                    = c | std::views::join;
    Container::difference_type n = 0;
    for (std::size_t i = 0; i < 10; ++i) {
      n += i;
      c.push_back(Container::value_type(i));
    }
    assert(std::distance(view.begin(), view.end()) == n);
  }
  {
    using Container = std::array<std::array<char, 3>, 10>;
    Container c;
    auto view = c | std::views::join;
    assert(std::distance(view.begin(), view.end()) == 30);
  }
  if (!TEST_IS_CONSTANT_EVALUATED) // TODO: Use TEST_STD_AT_LEAST_26_OR_RUNTIME_EVALUATED when std::deque is made constexpr
    test_deque();
#endif
  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER >= 17
  static_assert(tests(), "");
#endif
  return 0;
}
