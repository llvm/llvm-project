//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class indirect_array;

// T __get(size_t i);

#include <valarray>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  unsigned input[] = {0, 1, 2, 3, 4};
  const unsigned N = sizeof(input) / sizeof(input[0]);

  std::valarray<unsigned> array(input, N);

  {
    std::indirect_array<unsigned> result = array[std::valarray<std::size_t>(std::size_t(0), std::size_t(N))];
    for (unsigned i = 0; i < N; ++i)
      assert(result.__get(i) == 0);
  }

  {
    std::valarray<std::size_t> indirect(std::size_t(0), std::size_t(3));
    indirect[0]                          = 4;
    indirect[1]                          = 1;
    indirect[2]                          = 3;
    std::indirect_array<unsigned> result = array[indirect];
    assert(result.__get(0) == 4);
    assert(result.__get(1) == 1);
    assert(result.__get(2) == 3);
  }

  return 0;
}
