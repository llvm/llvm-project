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
// template<class U> simd(U&& value) noexcept;

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

template <class T, class SimdAbi, std::size_t array_size>
struct BroadCastHelper {
  const std::array<T, array_size>& expected_value;

  BroadCastHelper(const std::array<T, array_size>& value) : expected_value(value) {}

  template <class U>
  void operator()() const {
    if constexpr (is_non_narrowing_convertible_v<U, T>) {
      ex::simd<T, SimdAbi> simd_broadcast_from_vectorizable_type(static_cast<U>(3));
      assert_simd_values_equal<array_size>(simd_broadcast_from_vectorizable_type, expected_value);
    }
  }
};

template <class T, std::size_t>
struct CheckSimdBroadcastCtorFromVectorizedType {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<T, array_size> expected_value;
    std::fill(expected_value.begin(), expected_value.end(), 3);

    types::for_each(simd_test_types(), BroadCastHelper<T, SimdAbi, array_size>(expected_value));
  }
};

template <class T>
class implicit_type {
  T val;

public:
  implicit_type(T v) : val(v) {}
  operator T() const { return val; }
};

template <class T, std::size_t>
struct CheckSimdBroadcastCtor {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<T, array_size> expected_value;
    std::fill(expected_value.begin(), expected_value.end(), 3);

    implicit_type<T> implicit_convert_to_3(3);
    ex::simd<T, SimdAbi> simd_broadcast_from_implicit_type(std::move(implicit_convert_to_3));
    assert_simd_values_equal<array_size>(simd_broadcast_from_implicit_type, expected_value);

    ex::simd<T, SimdAbi> simd_broadcast_from_int(3);
    assert_simd_values_equal<array_size>(simd_broadcast_from_int, expected_value);

    if constexpr (std::is_unsigned_v<T>) {
      ex::simd<T, SimdAbi> simd_broadcast_from_uint(3u);
      assert_simd_values_equal<array_size>(simd_broadcast_from_uint, expected_value);
    }
  }
};

template <class T>
class no_implicit_type {
  T val;

public:
  no_implicit_type(T v) : val(v) {}
};

template <class U, class T, class SimdAbi = ex::simd_abi::compatible<T>, class = void>
struct has_broadcast_ctor : std::false_type {};

template <class U, class T, class SimdAbi>
struct has_broadcast_ctor<U, T, SimdAbi, std::void_t<decltype(ex::simd<T, SimdAbi>(std::declval<U>()))>>
    : std::true_type {};

template <class T, class SimdAbi>
struct CheckBroadcastCtorTraitsHelper {
  template <class U>
  void operator()() const {
    if constexpr (std::is_same_v<U, int>)
      static_assert(has_broadcast_ctor<U, T, SimdAbi>::value);
    else if constexpr (std::is_same_v<U, unsigned int> && std::is_unsigned_v<T>)
      static_assert(has_broadcast_ctor<U, T, SimdAbi>::value);
    else if constexpr (is_non_narrowing_convertible_v<U, T>)
      static_assert(has_broadcast_ctor<U, T, SimdAbi>::value);
    else
      static_assert(!has_broadcast_ctor<U, T, SimdAbi>::value);
  }
};

template <class T, std::size_t>
struct CheckBroadcastCtorTraits {
  template <class SimdAbi>
  void operator()() {
    types::for_each(simd_test_types(), CheckBroadcastCtorTraitsHelper<T, SimdAbi>());

    static_assert(!has_broadcast_ctor<no_implicit_type<T>, T, SimdAbi>::value);
    static_assert(has_broadcast_ctor<implicit_type<T>, T, SimdAbi>::value);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdBroadcastCtorFromVectorizedType>();
  test_all_simd_abi<CheckSimdBroadcastCtor>();
  test_all_simd_abi<CheckBroadcastCtorTraits>();
  return 0;
}
