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
// [simd.class]
// template<class Flags> simd_mask(const value_type* mem, Flags);

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdMaskLoadCtor {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;

    // element aligned tag
    bool element_buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      element_buffer[i] = static_cast<bool>(i % 2);
    ex::simd_mask<T, SimdAbi> element_mask(element_buffer, ex::element_aligned_tag());
    assert_simd_mask_values_equal(element_mask, element_buffer);

    // vector aligned tag
    alignas(ex::memory_alignment_v<ex::simd_mask<T, SimdAbi>>) bool vector_buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      vector_buffer[i] = static_cast<bool>(i % 2);
    ex::simd_mask<T, SimdAbi> vector_mask(vector_buffer, ex::vector_aligned_tag());
    assert_simd_mask_values_equal(vector_mask, vector_buffer);

    // overaligned tag
    alignas(bit_ceil(sizeof(bool) + 1)) bool overaligned_buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      overaligned_buffer[i] = static_cast<bool>(i % 2);
    ex::simd_mask<T, SimdAbi> overaligned_mask(overaligned_buffer, ex::overaligned_tag<bit_ceil(sizeof(bool) + 1)>());
    assert_simd_mask_values_equal(overaligned_mask, overaligned_buffer);
  }
};

template <class T, std::size_t>
struct CheckMaskLoadCtorTraits {
  template <class SimdAbi>
  void operator()() {
    // This function shall not participate in overload resolution unless
    // is_simd_flag_type_v<Flags> is true
    static_assert(std::is_constructible_v<ex::simd_mask<T, SimdAbi>, const bool*, ex::element_aligned_tag>);

    // is_simd_flag_type_v<Flags> is false
    static_assert(!std::is_constructible_v<ex::simd_mask<T, SimdAbi>, const bool*, T>);
    static_assert(!std::is_constructible_v<ex::simd_mask<T, SimdAbi>, const bool*, SimdAbi>);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdMaskLoadCtor>();
  test_all_simd_abi<CheckMaskLoadCtorTraits>();
  return 0;
}
