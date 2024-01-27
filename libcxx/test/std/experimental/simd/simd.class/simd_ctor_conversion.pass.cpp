//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: target=powerpc{{.*}}le-unknown-linux-gnu

// TODO: This test makes incorrect assumptions about floating point conversions.
//       See https://github.com/llvm/llvm-project/issues/74327.
// XFAIL: optimization=speed

// <experimental/simd>
//
// [simd.class]
// template<class U> simd(const simd<U, simd_abi::fixed_size<size()>>&) noexcept;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, class SimdAbi, std::size_t array_size>
struct ConversionHelper {
  const std::array<T, array_size>& expected_value;

  ConversionHelper(const std::array<T, array_size>& value) : expected_value(value) {}

  template <class U>
  void operator()() const {
    if constexpr (!std::is_same_v<U, T> && std::is_same_v<SimdAbi, ex::simd_abi::fixed_size<array_size>> &&
                  is_non_narrowing_convertible_v<U, T>) {
      static_assert(noexcept(ex::simd<T, SimdAbi>(ex::simd<U, SimdAbi>{})));
      ex::simd<U, SimdAbi> origin_simd([](U i) { return i; });
      ex::simd<T, SimdAbi> simd_from_implicit_conversion(origin_simd);
      assert_simd_values_equal<array_size>(simd_from_implicit_conversion, expected_value);
    }
  }
};

template <class T, std::size_t>
struct CheckConversionSimdCtor {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i);

    types::for_each(arithmetic_no_bool_types(), ConversionHelper<T, SimdAbi, array_size>(expected_value));
  }
};

template <class T, class SimdAbi, std::size_t array_size>
struct CheckConversionSimdCtorTraitsHelper {
  template <class U>
  void operator()() {
    if constexpr (!std::is_same_v<U, T>) {
      if constexpr (std::is_same_v<SimdAbi, ex::simd_abi::fixed_size<array_size>> &&
                    is_non_narrowing_convertible_v<U, T>)
        static_assert(std::is_convertible_v<ex::simd<U, SimdAbi>, ex::simd<T, SimdAbi>>);
      else
        static_assert(!std::is_convertible_v<ex::simd<U, SimdAbi>, ex::simd<T, SimdAbi>>);
    }
  }
};

template <class T, std::size_t>
struct CheckConversionSimdCtorTraits {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;

    types::for_each(arithmetic_no_bool_types(), CheckConversionSimdCtorTraitsHelper<T, SimdAbi, array_size>());
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckConversionSimdCtor>();
  test_all_simd_abi<CheckConversionSimdCtorTraits>();
  return 0;
}
