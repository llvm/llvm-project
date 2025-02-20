//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <simd>

// REQUIRES: std-at-least-c++26

#include <array>
#include <cassert>
#include <numeric>
#include <simd>
#include <type_traits>

#include "type_algorithms.h"
#include "../../utils.h"

namespace dp = std::datapar;

constexpr bool test() {
  types::for_each(types::vectorizable_types{}, []<class T> {
    simd_utils::test_sizes([]<int N>(std::integral_constant<int, N>) {
      std::array<T, N> arr;
      std::iota(std::begin(arr), std::end(arr), 0);
      const dp::simd<T, N> vec(arr); // make sure operator- is const
      std::same_as<dp::simd<T, N>> auto&& ret = -vec;
      for (int i = 0; i != N; ++i)
        assert(ret[i] == T(-i));
    });
  });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
