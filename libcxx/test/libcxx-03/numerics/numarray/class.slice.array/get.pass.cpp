//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class slice_array;

// T __get(size_t i);

#include <valarray>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  unsigned input[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const unsigned N = sizeof(input) / sizeof(input[0]);

  std::valarray<unsigned> array(input, N);

  {
    std::slice_array<unsigned> result = array[std::slice(0, N, 1)];
    for (unsigned i = 0; i < N; ++i)
      assert(result.__get(i) == i);
  }

  {
    std::slice_array<unsigned> result = array[std::slice(3, 2, 2)];
    assert(result.__get(0) == 3);
    assert(result.__get(1) == 5);
  }

  {
    std::slice_array<unsigned> result = array[std::slice(1, 3, 4)];
    assert(result.__get(0) == 1);
    assert(result.__get(1) == 5);
    assert(result.__get(2) == 9);
  }

  return 0;
}
