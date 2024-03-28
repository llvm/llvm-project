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
// [simd.reference]
// friend void swap(reference&& a, reference&& b) noexcept;
// friend void swap(value_type& a, reference&& b) noexcept;
// friend void swap(reference&& a, value_type& b) noexcept;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdRefSwap {
  template <class SimdAbi>
  void operator()() {
    ex::simd<T, SimdAbi> origin_simd_1(1);
    ex::simd<T, SimdAbi> origin_simd_2(2);
    T value = 3;

    static_assert(noexcept(swap(origin_simd_1[0], origin_simd_2[0])));
    swap(origin_simd_1[0], origin_simd_2[0]);
    assert((origin_simd_1[0] == 2) && (origin_simd_2[0] == 1));

    static_assert(noexcept(swap(origin_simd_1[0], value)));
    swap(origin_simd_1[0], value);
    assert((origin_simd_1[0] == 3) && (value == 2));

    static_assert(noexcept(swap(value, origin_simd_2[0])));
    swap(value, origin_simd_2[0]);
    assert((value == 1) && (origin_simd_2[0] == 2));
  }
};

template <class T, std::size_t>
struct CheckMaskRefSwap {
  template <class SimdAbi>
  void operator()() {
    ex::simd_mask<T, SimdAbi> origin_mask_1(true);
    ex::simd_mask<T, SimdAbi> origin_mask_2(false);
    bool value = true;

    static_assert(noexcept(swap(origin_mask_1[0], origin_mask_2[0])));
    swap(origin_mask_1[0], origin_mask_2[0]);
    assert((origin_mask_1[0] == false) && (origin_mask_2[0] == true));

    static_assert(noexcept(swap(origin_mask_1[0], value)));
    swap(origin_mask_1[0], value);
    assert((origin_mask_1[0] == true) && (value == false));

    static_assert(noexcept(swap(value, origin_mask_2[0])));
    swap(value, origin_mask_2[0]);
    assert((value == true) && (origin_mask_2[0] == false));
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdRefSwap>();
  test_all_simd_abi<CheckMaskRefSwap>();
  return 0;
}
