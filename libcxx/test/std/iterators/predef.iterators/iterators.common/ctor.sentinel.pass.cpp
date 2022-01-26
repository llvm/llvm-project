//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// constexpr common_iterator(S s);

#include <iterator>
#include <cassert>
#include <type_traits>

#include "test_iterators.h"

template<class It>
constexpr bool test() {
  using Sent = sentinel_wrapper<It>;
  using CommonIt = std::common_iterator<It, Sent>;
  int a[] = {1,2,3};
  It it = It(a);
  Sent sent = Sent(It(a+1));

  CommonIt lv = CommonIt(sent);
  assert(lv == CommonIt(sent));
  assert(lv != CommonIt(it));
  if (!std::is_constant_evaluated()) {
    assert(lv == std::next(CommonIt(it)));
  }

  CommonIt rv = CommonIt(std::move(sent));
  assert(rv == CommonIt(sent));
  assert(rv != CommonIt(it));
  if (!std::is_constant_evaluated()) {
    assert(rv == std::next(CommonIt(it)));
  }

  return true;
}

int main(int, char**) {
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();
  test<const int*>();

  static_assert(test<cpp17_input_iterator<int*>>());
  static_assert(test<forward_iterator<int*>>());
  static_assert(test<bidirectional_iterator<int*>>());
  static_assert(test<random_access_iterator<int*>>());
  static_assert(test<contiguous_iterator<int*>>());
  static_assert(test<int*>());
  static_assert(test<const int*>());

  return 0;
}
