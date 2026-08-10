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
// template <class T, class U = typename T::value_type> struct memory_alignment;
// template <class T, class U = typename T::value_type>
//   inline constexpr std::size_t memory_alignment_v = memory_alignment<T, U>::value;

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t N>
struct CheckMemoryAlignmentMask {
  template <class SimdAbi>
  void operator()() {
    LIBCPP_STATIC_ASSERT(
        ex::memory_alignment<ex::simd_mask<T, SimdAbi>>::value == bit_ceil(sizeof(bool) * ex::simd_size_v<T, SimdAbi>));
    LIBCPP_STATIC_ASSERT(
        ex::memory_alignment_v<ex::simd_mask<T, SimdAbi>> == bit_ceil(sizeof(bool) * ex::simd_size_v<T, SimdAbi>));
  }
};

template <class T, std::size_t N>
struct CheckMemoryAlignmentLongDouble {
  template <class SimdAbi>
  void operator()() {
    if constexpr (std::is_same_v<T, long double>) {
      // on i686-w64-mingw32-clang++, the size of long double is 12 bytes. Disambiguation is needed.
      static_assert(
          ex::memory_alignment<ex::simd<T, SimdAbi>>::value == bit_ceil(sizeof(T) * ex::simd_size_v<T, SimdAbi>));
      static_assert(ex::memory_alignment_v<ex::simd<T, SimdAbi>> == bit_ceil(sizeof(T) * ex::simd_size_v<T, SimdAbi>));
    }
  }
};

struct CheckMemAlignmentFixedDeduce {
  template <class T, std::size_t N>
  void check() {
    if constexpr (!std::is_same_v<T, long double>) {
      static_assert(ex::memory_alignment_v<ex::simd<T, ex::simd_abi::fixed_size<N>>> == sizeof(T) * bit_ceil(N),
                    "Memory Alignment mismatch with abi fixed_size");
      static_assert(ex::memory_alignment<ex::simd<T, ex::simd_abi::fixed_size<N>>>::value == sizeof(T) * bit_ceil(N),
                    "Memory Alignment mismatch with abi fixed_size");

      static_assert(ex::memory_alignment_v<ex::simd<T, ex::simd_abi::deduce_t<T, N>>> == sizeof(T) * bit_ceil(N),
                    "Memory Alignment mismatch with abi deduce");
      static_assert(ex::memory_alignment<ex::simd<T, ex::simd_abi::deduce_t<T, N>>>::value == sizeof(T) * bit_ceil(N),
                    "Memory Alignment mismatch with abi deduce");
    }
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

struct CheckMemAlignmentScalarNativeCompatible {
  template <class T>
  void operator()() {
    if constexpr (!std::is_same_v<T, long double>) {
      static_assert(ex::memory_alignment<ex::simd<T, ex::simd_abi::scalar>>::value == sizeof(T));
      static_assert(ex::memory_alignment_v<ex::simd<T, ex::simd_abi::scalar>> == sizeof(T));

      LIBCPP_STATIC_ASSERT(ex::memory_alignment<ex::simd<T, ex::simd_abi::compatible<T>>>::value == 16);
      LIBCPP_STATIC_ASSERT(ex::memory_alignment_v<ex::simd<T, ex::simd_abi::compatible<T>>> == 16);

      LIBCPP_STATIC_ASSERT(
          ex::memory_alignment<ex::simd<T, ex::simd_abi::native<T>>>::value == _LIBCPP_NATIVE_SIMD_WIDTH_IN_BYTES);
      LIBCPP_STATIC_ASSERT(
          ex::memory_alignment_v<ex::simd<T, ex::simd_abi::native<T>>> == _LIBCPP_NATIVE_SIMD_WIDTH_IN_BYTES);
    }
  }
};

template <class T, class U = typename T::value_type, class = void>
struct has_memory_alignment : std::false_type {};

template <class T, class U>
struct has_memory_alignment<T, U, std::void_t<decltype(ex::memory_alignment<T, U>::value)>> : std::true_type {};

struct CheckMemoryAlignmentTraits {
  template <class T>
  void operator()() {
    static_assert(has_memory_alignment<ex::native_simd<T>>::value);
    static_assert(has_memory_alignment<ex::fixed_size_simd_mask<T, 4>>::value);
    static_assert(has_memory_alignment<ex::native_simd<T>, T>::value);
    static_assert(has_memory_alignment<ex::fixed_size_simd_mask<T, 4>, bool>::value);

    static_assert(!has_memory_alignment<T, T>::value);
    static_assert(!has_memory_alignment<T, bool>::value);
    static_assert(!has_memory_alignment<ex::native_simd<T>, bool>::value);
    static_assert(!has_memory_alignment<ex::fixed_size_simd_mask<T, 4>, T>::value);
  }
};

int main(int, char**) {
  types::for_each(arithmetic_no_bool_types(), CheckMemAlignmentFixedDeduce());
  types::for_each(arithmetic_no_bool_types(), CheckMemAlignmentScalarNativeCompatible());
  test_all_simd_abi<CheckMemoryAlignmentMask>();
  test_all_simd_abi<CheckMemoryAlignmentLongDouble>();
  types::for_each(arithmetic_no_bool_types(), CheckMemoryAlignmentTraits());
  return 0;
}
