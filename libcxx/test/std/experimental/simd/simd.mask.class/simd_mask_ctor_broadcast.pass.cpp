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
// explicit simd_mask(value_type) noexcept;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckMaskBroadcastCtor {
  template <class SimdAbi>
  void operator()() {
    constexpr size_t array_size = ex::simd_size_v<T, SimdAbi>;
    const ex::simd_mask<T, SimdAbi> mask_ctor_from_broadcast(false);
    const std::array<bool, array_size> expected_value{};
    assert_simd_mask_values_equal(mask_ctor_from_broadcast, expected_value);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckMaskBroadcastCtor>();
  return 0;
}
