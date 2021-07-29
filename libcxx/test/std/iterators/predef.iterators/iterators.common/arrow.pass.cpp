//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// decltype(auto) operator->() const
//   requires see below;

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "types.h"

void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Case 2: http://eel.is/c++draft/iterators.common#common.iter.access-5.2
  {
    auto iter1 = simple_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    const auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);

    assert(commonIter1.operator->() == buffer);
    assert(commonIter2.operator->() == buffer);
  }

  // Case 3: http://eel.is/c++draft/iterators.common#common.iter.access-5.3
  {
    auto iter1 = value_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    const auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);

    assert(*commonIter1.operator->().operator->() == 1);
    assert(*commonIter2.operator->().operator->() == 1);
  }

  // Case 3: http://eel.is/c++draft/iterators.common#common.iter.access-5.3
  {
    auto iter1 = void_plus_plus_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    const auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);

    assert(*commonIter1.operator->().operator->() == 1);
    assert(*commonIter2.operator->().operator->() == 1);
  }

  // Case 1: http://eel.is/c++draft/iterators.common#common.iter.access-5.1
  {
    auto iter1 = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    const auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);

    assert(commonIter1.operator->().base() == buffer);
    assert(commonIter2.operator->().base() == buffer);
  }

  // Case 1: http://eel.is/c++draft/iterators.common#common.iter.access-5.1
  {
    auto iter1 = forward_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    const auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);

    assert(commonIter1.operator->().base() == buffer);
    assert(commonIter2.operator->().base() == buffer);
  }

  // Case 1: http://eel.is/c++draft/iterators.common#common.iter.access-5.1
  {
    auto iter1 = random_access_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    const auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);

    assert(commonIter1.operator->().base() == buffer);
    assert(commonIter2.operator->().base() == buffer);
  }
}

int main(int, char**) {
  test();

  return 0;
}
