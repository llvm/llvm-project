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
// template<class Flags> void copy_from(const value_type* mem, Flags);
// template<class Flags> void copy_to(value_type* mem, Flags);

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdMaskCopyFrom {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;

    // element aligned tag
    constexpr std::size_t element_alignas_size = alignof(bool);
    alignas(element_alignas_size) bool element_buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      element_buffer[i] = static_cast<bool>(i % 2);
    ex::simd_mask<T, SimdAbi> element_mask;
    element_mask.copy_from(element_buffer, ex::element_aligned_tag());
    assert_simd_mask_values_equal(element_mask, element_buffer);

    // vector aligned tag
    constexpr std::size_t vector_alignas_size = ex::memory_alignment_v<ex::simd_mask<T, SimdAbi>>;
    alignas(vector_alignas_size) bool vector_buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      vector_buffer[i] = static_cast<bool>(i % 2);
    ex::simd_mask<T, SimdAbi> vector_mask;
    vector_mask.copy_from(vector_buffer, ex::vector_aligned_tag());
    assert_simd_mask_values_equal(vector_mask, vector_buffer);

    // overaligned tag
    constexpr std::size_t over_alignas_size = bit_ceil(sizeof(bool) + 1);
    alignas(over_alignas_size) bool overaligned_buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      overaligned_buffer[i] = static_cast<bool>(i % 2);
    ex::simd_mask<T, SimdAbi> overaligned_mask;
    overaligned_mask.copy_from(overaligned_buffer, ex::overaligned_tag<over_alignas_size>());
    assert_simd_mask_values_equal(overaligned_mask, overaligned_buffer);
  }
};

template <class T, std::size_t>
struct CheckSimdMaskCopyTo {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;

    // element aligned tag
    constexpr std::size_t element_alignas_size = alignof(bool);
    alignas(element_alignas_size) bool element_buffer[array_size];
    ex::simd_mask<T, SimdAbi> element_mask(true);
    element_mask.copy_to(element_buffer, ex::element_aligned_tag());
    assert_simd_mask_values_equal(element_mask, element_buffer);

    // vector aligned tag
    constexpr std::size_t vector_alignas_size = ex::memory_alignment_v<ex::simd_mask<T, SimdAbi>>;
    alignas(vector_alignas_size) bool vector_buffer[array_size];
    ex::simd_mask<T, SimdAbi> vector_mask(false);
    vector_mask.copy_to(vector_buffer, ex::vector_aligned_tag());
    assert_simd_mask_values_equal(vector_mask, vector_buffer);

    // overaligned tag
    constexpr std::size_t over_alignas_size = bit_ceil(sizeof(bool) + 1);
    alignas(over_alignas_size) bool overaligned_buffer[array_size];
    ex::simd_mask<T, SimdAbi> overaligned_mask(true);
    overaligned_mask.copy_to(overaligned_buffer, ex::overaligned_tag<over_alignas_size>());
    assert_simd_mask_values_equal(overaligned_mask, overaligned_buffer);
  }
};

template <class T, class Flags, class SimdAbi = ex::simd_abi::compatible<T>, class = void>
struct has_copy_from : std::false_type {};

template <class T, class Flags, class SimdAbi>
struct has_copy_from<T,
                     Flags,
                     SimdAbi,
                     std::void_t<decltype(std::declval<ex::simd_mask<T, SimdAbi>>().copy_from(
                         std::declval<const bool*>(), std::declval<Flags>()))>> : std::true_type {};

template <class T, class Flags, class SimdAbi = ex::simd_abi::compatible<T>, class = void>
struct has_copy_to : std::false_type {};

template <class T, class Flags, class SimdAbi>
struct has_copy_to<T,
                   Flags,
                   SimdAbi,
                   std::void_t<decltype(std::declval<ex::simd_mask<T, SimdAbi>>().copy_to(
                       std::declval<bool*>(), std::declval<Flags>()))>> : std::true_type {};

template <class T, std::size_t>
struct CheckSimdMaskCopyTraits {
  template <class SimdAbi>
  void operator()() {
    // These functions shall not participate in overload resolution unless
    // is_simd_flag_type_v<Flags> is true
    static_assert(has_copy_from<T, ex::element_aligned_tag, SimdAbi>::value);
    static_assert(has_copy_to<T, ex::element_aligned_tag, SimdAbi>::value);

    // is_simd_flag_type_v<Flags> is false
    static_assert(!has_copy_from<T, T, SimdAbi>::value);
    static_assert(!has_copy_to<T, T, SimdAbi>::value);
    static_assert(!has_copy_from<T, SimdAbi, SimdAbi>::value);
    static_assert(!has_copy_to<T, SimdAbi, SimdAbi>::value);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdMaskCopyFrom>();
  test_all_simd_abi<CheckSimdMaskCopyTo>();
  test_all_simd_abi<CheckSimdMaskCopyTraits>();
  return 0;
}
