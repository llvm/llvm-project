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
// [simd.traits]
//template <class T, class Abi = simd_abi::compatible<T>> struct simd_size;
//template <class T, class Abi = simd_abi::compatible<T>>
//inline constexpr std::size_t ex::simd_size_v = ex::simd_size<T, Abi>::value;

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

struct CheckSimdSizeFixedDeduce {
  template <class T, std::size_t N>
  void check() {
    static_assert(ex::simd_size_v<T, ex::simd_abi::fixed_size<N>> == N, "Simd size mismatch with abi fixed_size");
    static_assert(ex::simd_size<T, ex::simd_abi::fixed_size<N>>::value == N, "Simd size mismatch with abi fixed_size");

    static_assert(ex::simd_size_v<T, ex::simd_abi::deduce_t<T, N>> == N, "Simd size mismatch with abi deduce");
    static_assert(ex::simd_size<T, ex::simd_abi::deduce_t<T, N>>::value == N, "Simd size mismatch with abi deduce");
  }

  template <class T, std::size_t... N>
  void performChecks(std::index_sequence<N...>) {
    (check<T, N + 1>(), ...);
  }

  template <class T>
  void operator()() {
    performChecks<T>(std::make_index_sequence<max_simd_size>{});
  }
};

struct CheckSimdSizeScalarNativeCompatible {
  template <class T>
  void operator()() {
    static_assert(ex::simd_size_v<T, ex::simd_abi::scalar> == 1);
    static_assert(ex::simd_size<T, ex::simd_abi::scalar>::value == 1);

    LIBCPP_STATIC_ASSERT(ex::simd_size<T, ex::simd_abi::compatible<T>>::value == 16 / sizeof(T));
    LIBCPP_STATIC_ASSERT(ex::simd_size_v<T, ex::simd_abi::compatible<T>> == 16 / sizeof(T));

    LIBCPP_STATIC_ASSERT(
        ex::simd_size<T, ex::simd_abi::native<T>>::value == _LIBCPP_NATIVE_SIMD_WIDTH_IN_BYTES / sizeof(T));
    LIBCPP_STATIC_ASSERT(ex::simd_size_v<T, ex::simd_abi::native<T>> == _LIBCPP_NATIVE_SIMD_WIDTH_IN_BYTES / sizeof(T));
  }
};

template <class T, class Abi = ex::simd_abi::compatible<T>, class = void>
struct has_simd_size : std::false_type {};

template <class T, class Abi>
struct has_simd_size<T, Abi, std::void_t<decltype(ex::simd_size<T, Abi>::value)>> : std::true_type {};

struct CheckSimdSizeTraits {
  template <class T>
  void operator()() {
    static_assert(has_simd_size<T>::value);
    static_assert(!has_simd_size<ex::native_simd<T>>::value);

    static_assert(has_simd_size<T, ex::simd_abi::scalar>::value);
    static_assert(has_simd_size<T, ex::simd_abi::fixed_size<4>>::value);
    static_assert(has_simd_size<T, ex::simd_abi::native<T>>::value);
    static_assert(has_simd_size<T, ex::simd_abi::compatible<T>>::value);

    static_assert(!has_simd_size<ex::simd_abi::native<T>, ex::simd_abi::native<T>>::value);
    static_assert(!has_simd_size<ex::native_simd<T>, ex::simd_abi::native<T>>::value);
    static_assert(!has_simd_size<ex::fixed_size_simd<T, 3>, ex::simd_abi::native<T>>::value);
    static_assert(!has_simd_size<ex::fixed_size_simd_mask<T, 4>, ex::simd_abi::native<T>>::value);

    static_assert(!has_simd_size<T, T>::value);
    static_assert(!has_simd_size<T, ex::native_simd<T>>::value);
    static_assert(!has_simd_size<T, ex::fixed_size_simd<T, 3>>::value);
    static_assert(!has_simd_size<T, ex::fixed_size_simd_mask<T, 4>>::value);
  }
};

int main(int, char**) {
  types::for_each(arithmetic_no_bool_types(), CheckSimdSizeFixedDeduce());
  types::for_each(arithmetic_no_bool_types(), CheckSimdSizeScalarNativeCompatible());
  types::for_each(arithmetic_no_bool_types(), CheckSimdSizeTraits());
  return 0;
}
