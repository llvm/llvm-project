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
// template<class U> simd_mask(const simd_mask<U, simd_abi::fixed_size<size()>>&) noexcept;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, class SimdAbi, std::size_t array_size>
struct ConversionHelper {
  const std::array<bool, array_size>& expected_value;

  ConversionHelper(const std::array<bool, array_size>& value) : expected_value(value) {}

  template <class U>
  void operator()() const {
    if constexpr (!std::is_same_v<U, T> && std::is_same_v<SimdAbi, ex::simd_abi::fixed_size<array_size>>) {
      static_assert(noexcept(ex::simd_mask<T, SimdAbi>(ex::simd_mask<U, SimdAbi>{})));
      ex::simd_mask<U, SimdAbi> origin_mask(false);
      ex::simd_mask<T, SimdAbi> mask_from_implicit_conversion(origin_mask);
      assert_simd_mask_values_equal<array_size>(mask_from_implicit_conversion, expected_value);
    }
  }
};

template <class T, std::size_t>
struct CheckConversionMaskCtor {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<bool, array_size> expected_value{};

    types::for_each(simd_test_types(), ConversionHelper<T, SimdAbi, array_size>(expected_value));
  }
};

template <class T, class SimdAbi, std::size_t array_size>
struct CheckConversionMaskCtorTraitsHelper {
  template <class U>
  void operator()() {
    if constexpr (!std::is_same_v<U, T>) {
      if constexpr (std::is_same_v<SimdAbi, ex::simd_abi::fixed_size<array_size>>)
        static_assert(std::is_convertible_v<ex::simd_mask<U, SimdAbi>, ex::simd_mask<T, SimdAbi>>);
      else
        static_assert(!std::is_convertible_v<ex::simd_mask<U, SimdAbi>, ex::simd_mask<T, SimdAbi>>);
    }
  }
};

template <class T, std::size_t>
struct CheckConversionMaskCtorTraits {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;

    types::for_each(simd_test_types(), CheckConversionMaskCtorTraitsHelper<T, SimdAbi, array_size>());
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckConversionMaskCtor>();
  test_all_simd_abi<CheckConversionMaskCtorTraits>();
  return 0;
}
