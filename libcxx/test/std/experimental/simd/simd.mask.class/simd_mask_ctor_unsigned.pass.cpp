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
// template<class U> constexpr explicit simd_mask(U val) noexcept;

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdMaskCtorUnsigned {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    
    // Pattern: 101010...
    unsigned long long pattern = 0xAAAAAAAAAAAAAAAA; 
    bool expected_buffer[array_size];

    constexpr size_t limit = (array_size < 64) ? array_size : 64;

    for (size_t i = 0; i < limit; ++i) {
      expected_buffer[i] = (pattern >> i) & 1;
    }


    for (size_t i = limit; i < array_size; ++i) {
      expected_buffer[i] = false;
    }

    ex::simd_mask<T, SimdAbi> mask(pattern);

    assert_simd_mask_values_equal(mask, expected_buffer);
    static_assert(noexcept(ex::simd_mask<T, SimdAbi>(pattern)));
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdMaskCtorUnsigned>();
  return 0;
}