//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class mask_array;

// T __get(size_t i);

#include <valarray>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  unsigned input[] = {0, 1, 2, 3, 4};
  const unsigned N = sizeof(input) / sizeof(input[0]);

  std::valarray<unsigned> array(input, N);

  {
    std::mask_array<unsigned> result = array[std::valarray<bool>(true, N)];
    for (unsigned i = 0; i < N; ++i)
      assert(result.__get(i) == i);
  }

  {
    std::valarray<bool> mask(false, N);
    mask[1]                          = true;
    mask[3]                          = true;
    std::mask_array<unsigned> result = array[mask];
    assert(result.__get(0) == 1);
    assert(result.__get(1) == 3);
  }

  {
    std::valarray<bool> mask(false, N);
    mask[0]                          = true;
    mask[4]                          = true;
    std::mask_array<unsigned> result = array[mask];
    assert(result.__get(0) == 0);
    assert(result.__get(1) == 4);
  }

  return 0;
}
