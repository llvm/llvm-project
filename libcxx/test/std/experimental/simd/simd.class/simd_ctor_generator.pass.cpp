//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: target=powerpc{{.*}}le-unknown-linux-gnu

// <experimental/simd>
//
// [simd.class]
// template<class G> explicit simd(G&& gen) noexcept;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckGenerateSimdCtor {
  template <class SimdAbi>
  void operator()() {
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    constexpr size_t array_size = origin_simd.size();
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i);
    assert_simd_values_equal<array_size>(origin_simd, expected_value);
  }
};

template <class U, class T, class SimdAbi = ex::simd_abi::compatible<T>, class = void>
struct has_generate_ctor : std::false_type {};

template <class U, class T, class SimdAbi>
struct has_generate_ctor<U, T, SimdAbi, std::void_t<decltype(ex::simd<T, SimdAbi>(std::declval<U>()))>>
    : std::true_type {};

template <class T, std::size_t>
struct CheckGenerateCtorTraits {
  template <class SimdAbi>
  void operator()() {
    static_assert(!has_generate_ctor<SimdAbi, T, SimdAbi>::value);

    auto func = [](T i) { return i; };
    static_assert(has_generate_ctor<decltype(func), T, SimdAbi>::value);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckGenerateSimdCtor>();
  test_all_simd_abi<CheckGenerateCtorTraits>();
  return 0;
}
