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
// template<class U, class Flags> simd(const U* mem, Flags);

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

template <class T, class SimdAbi, std::size_t array_size>
struct ElementAlignedLoadCtorHelper {
  template <class U>
  void operator()() const {
    U buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      buffer[i] = static_cast<U>(i);
    ex::simd<T, SimdAbi> origin_simd(buffer, ex::element_aligned_tag());
    assert_simd_values_equal(origin_simd, buffer);
  }
};

template <class T, class SimdAbi, std::size_t array_size>
struct VectorAlignedLoadCtorHelper {
  template <class U>
  void operator()() const {
    alignas(ex::memory_alignment_v<ex::simd<T, SimdAbi>, U>) U buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      buffer[i] = static_cast<U>(i);
    ex::simd<T, SimdAbi> origin_simd(buffer, ex::vector_aligned_tag());
    assert_simd_values_equal(origin_simd, buffer);
  }
};

template <class T, class SimdAbi, std::size_t array_size>
struct OveralignedLoadCtorHelper {
  template <class U>
  void operator()() const {
    alignas(bit_ceil(sizeof(U) + 1)) U buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      buffer[i] = static_cast<U>(i);
    ex::simd<T, SimdAbi> origin_simd(buffer, ex::overaligned_tag<bit_ceil(sizeof(U) + 1)>());
    assert_simd_values_equal(origin_simd, buffer);
  }
};

template <class T, std::size_t>
struct CheckSimdLoadCtor {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;

    types::for_each(arithmetic_no_bool_types(), ElementAlignedLoadCtorHelper<T, SimdAbi, array_size>());
    types::for_each(arithmetic_no_bool_types(), VectorAlignedLoadCtorHelper<T, SimdAbi, array_size>());
    types::for_each(arithmetic_no_bool_types(), OveralignedLoadCtorHelper<T, SimdAbi, array_size>());
  }
};

template <class T, std::size_t>
struct CheckLoadCtorTraits {
  template <class SimdAbi>
  void operator()() {
    // This function shall not participate in overload resolution unless
    // is_simd_flag_type_v<Flags> is true, and
    // U is a vectorizable type.
    static_assert(std::is_constructible_v<ex::simd<T, SimdAbi>, const int*, ex::element_aligned_tag>);

    // is_simd_flag_type_v<Flags> is false
    static_assert(!std::is_constructible_v<ex::simd<T, SimdAbi>, const int*, T>);
    static_assert(!std::is_constructible_v<ex::simd<T, SimdAbi>, const int*, SimdAbi>);

    // U is not a vectorizable type.
    static_assert(!std::is_constructible_v<ex::simd<T, SimdAbi>, const SimdAbi*, ex::element_aligned_tag>);
    static_assert(
        !std::is_constructible_v<ex::simd<T, SimdAbi>, const ex::element_aligned_tag*, ex::element_aligned_tag>);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdLoadCtor>();
  test_all_simd_abi<CheckLoadCtorTraits>();
  return 0;
}
