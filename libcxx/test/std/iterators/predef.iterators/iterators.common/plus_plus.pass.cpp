//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// common_iterator& operator++();
// decltype(auto) operator++(int);

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "types.h"

struct Incomplete;

void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Reference: http://eel.is/c++draft/iterators.common#common.iter.nav-5
  // Case 2: can-reference
  {
    auto iter1 = simple_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
  }

  // Case 2: can-reference
  {
    auto iter1 = value_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
  }

  // Case 3: postfix-proxy
  {
    auto iter1 = void_plus_plus_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
  }

  // Case 2: where this is not referencable or move constructible
  {
    auto iter1 = value_type_not_move_constructible_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    commonIter1++;
    // Note: postfix operator++ returns void.
    // assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i) {
      assert(*commonIter1 == i);
      commonIter1++;
    }
    assert(commonIter1 == commonSent1);
  }

  // Case 2: can-reference
  {
    auto iter1 = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
  }

  // Case 1: forward_iterator
  {
    auto iter1 = forward_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
  }

  // Case 1: forward_iterator
  {
    auto iter1 = random_access_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
  }
}

int main(int, char**) {
  test();

  return 0;
}
