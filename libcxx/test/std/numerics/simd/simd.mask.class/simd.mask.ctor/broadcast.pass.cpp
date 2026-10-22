//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <simd>

// REQUIRES: std-at-least-c++26

#include <cassert>
#include <simd>
#include <type_traits>

#include "type_algorithms.h"
#include "../../utils.h"

namespace dp = std::datapar;

constexpr bool test() {
  types::for_each(types::vectorizable_types{}, []<class T>() {
    {
      dp::simd_mask<T> vec(true);
      for (auto i = 0; i != vec.size(); ++i)
        assert(vec[i]);
    }
    {
      dp::simd_mask<T> vec(false);
      for (auto i = 0; i != vec.size(); ++i)
        assert(!vec[i]);
    }
  });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
