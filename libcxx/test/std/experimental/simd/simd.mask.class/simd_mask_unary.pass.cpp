//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <experimental/simd>
//
// [simd.mask.class]
// simd_mask operator!() const noexcept;

#include "../test_utils.h"
#include <algorithm>
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdMaskNotOperator {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd_mask<T, SimdAbi> origin_mask(true);
    static_assert(noexcept(!origin_mask));
    std::array<bool, array_size> expected_value;
    std::fill(expected_value.begin(), expected_value.end(), false);
    assert_simd_mask_values_equal<array_size>(!origin_mask, expected_value);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdMaskNotOperator>();
  return 0;
}
