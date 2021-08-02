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

// constexpr decltype(auto) operator*();
// constexpr decltype(auto) operator*() const
//   requires dereferenceable<const I>;

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

struct InputOrOutputArchetype {
  using difference_type = int;

  int *ptr;

  constexpr int operator*() const { return *ptr; }
  constexpr void operator++(int) { ++ptr; }
  constexpr InputOrOutputArchetype& operator++() { ++ptr; return *this; }
};

struct NonConstDeref {
  using difference_type = int;

  int *ptr;

  constexpr int operator*() { return *ptr; }
  constexpr void operator++(int) { ++ptr; }
  constexpr NonConstDeref& operator++() { ++ptr; return *this; }
};

template<class T>
concept IsDereferenceable = requires(T& i) {
  *i;
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert( IsDereferenceable<std::counted_iterator<InputOrOutputArchetype>>);
    static_assert( IsDereferenceable<const std::counted_iterator<InputOrOutputArchetype>>);
    static_assert( IsDereferenceable<std::counted_iterator<NonConstDeref>>);
    static_assert(!IsDereferenceable<const std::counted_iterator<NonConstDeref>>);
  }

  {
    std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    for (int i = 1; i < 9; ++i, ++iter)
      assert(*iter == i);
  }

  {
    std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);
    for (int i = 1; i < 9; ++i, ++iter)
      assert(*iter == i);
  }

  {
    std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    for (int i = 1; i < 9; ++i, ++iter)
      assert(*iter == i);
  }

  {
    std::counted_iterator iter(InputOrOutputArchetype{buffer}, 8);
    for (int i = 1; i < 9; ++i, ++iter)
      assert(*iter == i);
  }

  {
    const std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(*iter == 1);
  }

  {
    const std::counted_iterator iter(forward_iterator<int*>{buffer + 1}, 7);
    assert(*iter == 2);
  }

  {
    const std::counted_iterator iter(contiguous_iterator<int*>{buffer + 2}, 6);
    assert(*iter == 3);
  }

  {
    const std::counted_iterator iter(InputOrOutputArchetype{buffer + 2}, 6);
    assert(*iter == 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
