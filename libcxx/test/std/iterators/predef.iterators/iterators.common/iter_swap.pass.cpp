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

// template<indirectly_swappable<I> I2, class S2>
//   friend void iter_swap(const common_iterator& x, const common_iterator<I2, S2>& y)
//     noexcept(noexcept(ranges::iter_swap(declval<const I&>(), declval<const I2&>())));

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "types.h"

void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto iter1 = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    for (auto i = 0; i < 4; ++i) ++commonIter2;
    assert(*commonIter2 == 5);
    std::ranges::iter_swap(commonIter1, commonIter2);
    assert(*commonIter1 == 5);
    assert(*commonIter2 == 1);
    std::ranges::iter_swap(commonIter2, commonIter1);
  }
  {
    auto iter1 = forward_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    for (auto i = 0; i < 4; ++i) ++commonIter2;
    assert(*commonIter2 == 5);
    std::ranges::iter_swap(commonIter1, commonIter2);
    assert(*commonIter1 == 5);
    assert(*commonIter2 == 1);
    std::ranges::iter_swap(commonIter2, commonIter1);
  }
  {
    auto iter1 = random_access_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonIter2 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    for (auto i = 0; i < 4; ++i) ++commonIter2;
    assert(*commonIter2 == 5);
    std::ranges::iter_swap(commonIter1, commonIter2);
    assert(*commonIter1 == 5);
    assert(*commonIter2 == 1);
    std::ranges::iter_swap(commonIter2, commonIter1);
  }
}

int main(int, char**) {
  test();

  return 0;
}
