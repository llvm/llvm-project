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
// [simd.mask.ctor]
// constexpr simd_mask(const bitset<size()>& b) noexcept;

#include "../test_utils.h"
#include <bitset>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdMaskCtorBitset {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    
    std::bitset<array_size> b;
    bool expected_buffer[array_size];

    for (size_t i = 0; i < array_size; ++i) {
      bool val = (i % 2 == 0);
      if (val) 
        b[i] = true;
      expected_buffer[i] = val;
    }

    // Construct mask
    ex::simd_mask<T, SimdAbi> mask(b);

    assert_simd_mask_values_equal(mask, expected_buffer);
    static_assert(noexcept(ex::simd_mask<T, SimdAbi>(b)));
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdMaskCtorBitset>();
  return 0;
}