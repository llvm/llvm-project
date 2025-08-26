//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class gslice_array;

// T __get(size_t i);

#include <valarray>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  unsigned input[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const unsigned N = sizeof(input) / sizeof(input[0]);

  std::valarray<unsigned> array(input, N);

  {
    std::gslice_array<unsigned> result =
        array[std::gslice(0, std::valarray<std::size_t>(N, 1), std::valarray<std::size_t>(1, 1))];
    for (unsigned i = 0; i < N; ++i)
      assert(result.__get(i) == i);
  }

  {
    std::valarray<std::size_t> sizes(2);
    sizes[0] = 2;
    sizes[1] = 3;

    std::valarray<std::size_t> strides(2);
    strides[0] = 6;
    strides[1] = 1;

    std::gslice_array<unsigned> result = array[std::gslice(1, sizes, strides)];
    assert(result.__get(0) == input[1 + 0 * 6 + 0 * 1]);
    assert(result.__get(1) == input[1 + 0 * 6 + 1 * 1]);
    assert(result.__get(2) == input[1 + 0 * 6 + 2 * 1]);

    assert(result.__get(3) == input[1 + 1 * 6 + 0 * 1]);
    assert(result.__get(4) == input[1 + 1 * 6 + 1 * 1]);
    assert(result.__get(5) == input[1 + 1 * 6 + 2 * 1]);
  }
  return 0;
}
