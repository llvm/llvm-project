//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// constexpr iter_difference_t<I> count() const noexcept;

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

struct InputOrOutputArchetype {
  using difference_type = int;

  int *ptr;

  constexpr int operator*() { return *ptr; }
  constexpr void operator++(int) { ++ptr; }
  constexpr InputOrOutputArchetype& operator++() { ++ptr; return *this; }
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    for (int i = 8; i != 0; --i, ++iter)
      assert(iter.count() == i);

    static_assert(noexcept(iter.count()));
  }
  {
    std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);
    for (int i = 8; i != 0; --i, ++iter)
      assert(iter.count() == i);

    static_assert(noexcept(iter.count()));
  }
  {
    std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    for (int i = 8; i != 0; --i, ++iter)
      assert(iter.count() == i);
  }
  {
    std::counted_iterator iter(InputOrOutputArchetype{buffer + 2}, 6);
    assert(iter.count() == 6);
  }

  // Const tests.
  {
    const std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(iter.count() == 8);
  }
  {
    const std::counted_iterator iter(forward_iterator<int*>{buffer + 1}, 7);
    assert(iter.count() == 7);
  }
  {
    const std::counted_iterator iter(contiguous_iterator<int*>{buffer + 2}, 6);
    assert(iter.count() == 6);
  }
  {
    const std::counted_iterator iter(InputOrOutputArchetype{buffer + 2}, 6);
    assert(iter.count() == 6);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
