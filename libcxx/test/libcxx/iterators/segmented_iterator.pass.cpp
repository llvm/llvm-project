//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <iterator>

// __segmented_iterator_traits<_Iter>

// verifies that __segmented_iterator_traits<_Iter> does not result in implicit
// template instantaition, which may cause hard errors in SFINAE.

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <array>
#include <deque>
#include <list>
#include <ranges>
#include <vector>
#include <__iterator/segmented_iterator.h>
#include <__type_traits/integral_constant.h>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
struct is_segmented_random_access_iterator
    : std::_BoolConstant<std::__is_segmented_iterator<Iter>::value &&
                         std::__has_random_access_iterator_category<
                             typename std::__segmented_iterator_traits<Iter>::__local_iterator>::value> {};

int main(int, char**) {
  static_assert(is_segmented_random_access_iterator<std::deque<int>::iterator>::value, "");
  static_assert(!is_segmented_random_access_iterator<std::vector<int>::iterator>::value, "");
  static_assert(!is_segmented_random_access_iterator<std::list<int>::iterator>::value, "");
  static_assert(!is_segmented_random_access_iterator<std::array<int, 0>::iterator>::value, "");
  static_assert(!is_segmented_random_access_iterator<cpp17_input_iterator<int*> >::value, "");
  static_assert(!is_segmented_random_access_iterator<forward_iterator<int*> >::value, "");
  static_assert(!is_segmented_random_access_iterator<random_access_iterator<int*> >::value, "");
  static_assert(!is_segmented_random_access_iterator<int*>::value, "");

#if TEST_STD_VER >= 20
  using join_view_iterator = decltype((std::declval<std::vector<std::vector<int > >&>() | std::views::join).begin());
  static_assert(is_segmented_random_access_iterator<join_view_iterator>::value, "");
#endif

  return 0;
}
