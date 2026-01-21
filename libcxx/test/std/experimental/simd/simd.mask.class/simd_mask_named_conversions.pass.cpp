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
// [simd.mask.namedconv]
// constexpr bitset<size()> to_bitset() const noexcept;
// constexpr unsigned long long to_ullong() const;

#include "../test_utils.h"
#include <bitset>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdMaskToBitset {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;

    // Case: All True
    ex::simd_mask<T, SimdAbi> mask(true);
    std::bitset<array_size> b = mask.to_bitset();
    assert(b.all());

    // Case: All False
    ex::simd_mask<T, SimdAbi> mask_false(false);
    std::bitset<array_size> b_false = mask_false.to_bitset();
    assert(b_false.none());

    static_assert(noexcept(mask.to_bitset()));
  }
};

template <class T, std::size_t>
struct CheckSimdMaskToUllong {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;

    // Case: All True
    ex::simd_mask<T, SimdAbi> mask(true);
    unsigned long long val = mask.to_ullong();

    unsigned long long expected = ~0ULL;
    if (array_size < 64) {
      expected = (1ULL << array_size) - 1;
    }

    assert(val == expected);

    // Case: All False
    ex::simd_mask<T, SimdAbi> mask_false(false);
    assert(mask_false.to_ullong() == 0ULL);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdMaskToBitset>();
  test_all_simd_abi<CheckSimdMaskToUllong>();
  return 0;
}