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

template <class T>
constexpr void test() {
  simd_utils::test_sizes([]<int N>(std::integral_constant<int, N>) {
    std::array<T, N> arr;
    std::iota(std::begin(arr), std::end(arr), 0);
    dp::simd<T, N> vec(arr);
    for (auto i = 0; i != vec.size(); ++i)
      assert(vec[i] == T(i));
  });
}

constexpr bool test() {
  types::for_each(types::vectorizable_types{}, []<class T> { test<T>(); });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
