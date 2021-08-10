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

// template<class I2, sentinel_for<I> S2>
//   requires sentinel_for<S, I2>
// friend bool operator==(
//   const common_iterator& x, const common_iterator<I2, S2>& y);
// template<class I2, sentinel_for<I> S2>
//   requires sentinel_for<S, I2> && equality_comparable_with<I, I2>
// friend bool operator==(
//   const common_iterator& x, const common_iterator<I2, S2>& y);

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "types.h"

void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto iter1 = simple_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    const auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    const auto commonSent2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(commonIter1 != commonSent1);
    assert(commonIter2 != commonSent2);
    assert(commonSent1 != commonIter1);
    assert(commonSent2 != commonIter2);

    for (auto i = 1; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
    assert(commonSent1 == commonIter1);
  }
  {
    auto iter1 = value_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    const auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    const auto commonSent2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(commonIter1 != commonSent1);
    assert(commonIter2 != commonSent2);
    assert(commonSent1 != commonIter1);
    assert(commonSent2 != commonIter2);

    for (auto i = 1; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
    assert(commonSent1 == commonIter1);
  }
  {
    auto iter1 = simple_iterator<int*>(buffer);
    auto iter2 = comparable_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    const auto commonIter2 = std::common_iterator<decltype(iter2), sentinel_type<int*>>(iter2);
    const auto commonSent2 = std::common_iterator<decltype(iter2), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(commonIter1 == commonIter2);
    assert(commonSent1 != commonIter2);
    assert(commonSent1 == commonSent2);
    assert(commonSent2 == commonSent1);

    assert(commonIter1 != commonSent1);
    assert(commonIter2 != commonSent2);
    assert(commonSent1 != commonIter1);
    assert(commonSent2 != commonIter2);

    assert(commonIter1 == commonIter2);
    assert(commonIter2 == commonIter1);

    for (auto i = 1; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
    assert(commonSent1 == commonIter1);

    // This check may *seem* incorrect (our iterators point to two completely different
    // elements of buffer). However, this is actually what the Standard wants.
    // See https://eel.is/c++draft/iterators.common#common.iter.cmp-2.
    assert(commonIter1 == commonIter2);
  }
  {
    auto iter1 = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    const auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    const auto commonSent2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(commonIter1 != commonSent1);
    assert(commonIter2 != commonSent2);
    assert(commonSent1 != commonIter1);
    assert(commonSent2 != commonIter2);

    for (auto i = 1; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
    assert(commonSent1 == commonIter1);
  }
  {
    auto iter1 = forward_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    const auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    const auto commonSent2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(commonIter1 != commonSent1);
    assert(commonIter2 != commonSent2);
    assert(commonSent1 != commonIter1);
    assert(commonSent2 != commonIter2);

    for (auto i = 1; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
    assert(commonSent1 == commonIter1);
  }
  {
    auto iter1 = random_access_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    const auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    const auto commonSent2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(commonIter1 != commonSent1);
    assert(commonIter2 != commonSent2);
    assert(commonSent1 != commonIter1);
    assert(commonSent2 != commonIter2);

    assert(commonSent1 == commonSent2);
    assert(commonSent2 == commonSent1);

    for (auto i = 1; commonIter1 != commonSent1; ++i) {
      assert(*(commonIter1++) == i);
    }
    assert(commonIter1 == commonSent1);
    assert(commonSent1 == commonIter1);
  }
}

int main(int, char**) {
  test();

  return 0;
}
