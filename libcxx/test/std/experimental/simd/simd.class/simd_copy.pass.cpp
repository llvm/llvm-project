//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// FIXME: Fatal error with following targets (remove XFAIL when fixed):
//   Pass-by-value arguments with alignment greater than register width are not supported.
// XFAIL: target=powerpc{{.*}}-ibm-aix7.2.5.7

// <experimental/simd>
//
// [simd.class]
// template<class U, class Flags> void copy_from(const U* mem, Flags);
// template<class U, class Flags> void copy_to(U* mem, Flags) const;

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

template <class T, class SimdAbi, std::size_t array_size>
struct ElementAlignedCopyFromHelper {
  template <class U>
  void operator()() const {
    U buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      buffer[i] = static_cast<U>(i);
    ex::simd<T, SimdAbi> origin_simd;
    origin_simd.copy_from(buffer, ex::element_aligned_tag());
    assert_simd_values_equal(origin_simd, buffer);
  }
};

template <class T, class SimdAbi, std::size_t array_size>
struct VectorAlignedCopyFromHelper {
  template <class U>
  void operator()() const {
    alignas(ex::memory_alignment_v<ex::simd<T, SimdAbi>, U>) U buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      buffer[i] = static_cast<U>(i);
    ex::simd<T, SimdAbi> origin_simd;
    origin_simd.copy_from(buffer, ex::vector_aligned_tag());
    assert_simd_values_equal(origin_simd, buffer);
  }
};

template <class T, class SimdAbi, std::size_t array_size>
struct OveralignedCopyFromHelper {
  template <class U>
  void operator()() const {
    alignas(bit_ceil(sizeof(U) + 1)) U buffer[array_size];
    for (size_t i = 0; i < array_size; ++i)
      buffer[i] = static_cast<U>(i);
    ex::simd<T, SimdAbi> origin_simd;
    origin_simd.copy_from(buffer, ex::overaligned_tag<bit_ceil(sizeof(U) + 1)>());
    assert_simd_values_equal(origin_simd, buffer);
  }
};

template <class T, std::size_t>
struct CheckSimdCopyFrom {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;

    types::for_each(simd_test_types(), ElementAlignedCopyFromHelper<T, SimdAbi, array_size>());
    types::for_each(simd_test_types(), VectorAlignedCopyFromHelper<T, SimdAbi, array_size>());
    types::for_each(simd_test_types(), OveralignedCopyFromHelper<T, SimdAbi, array_size>());
  }
};

template <class T, class SimdAbi, std::size_t array_size>
struct ElementAlignedCopyToHelper {
  template <class U>
  void operator()() const {
    U buffer[array_size];
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    origin_simd.copy_to(buffer, ex::element_aligned_tag());
    assert_simd_values_equal(origin_simd, buffer);
  }
};

template <class T, class SimdAbi, std::size_t array_size>
struct VectorAlignedCopyToHelper {
  template <class U>
  void operator()() const {
    alignas(ex::memory_alignment_v<ex::simd<T, SimdAbi>, U>) U buffer[array_size];
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    origin_simd.copy_to(buffer, ex::vector_aligned_tag());
    assert_simd_values_equal(origin_simd, buffer);
  }
};

template <class T, class SimdAbi, std::size_t array_size>
struct OveralignedCopyToHelper {
  template <class U>
  void operator()() const {
    alignas(bit_ceil(sizeof(U) + 1)) U buffer[array_size];
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    origin_simd.copy_to(buffer, ex::overaligned_tag<bit_ceil(sizeof(U) + 1)>());
    assert_simd_values_equal(origin_simd, buffer);
  }
};

template <class T, std::size_t>
struct CheckSimdCopyTo {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;

    types::for_each(simd_test_types(), ElementAlignedCopyToHelper<T, SimdAbi, array_size>());
    types::for_each(simd_test_types(), VectorAlignedCopyToHelper<T, SimdAbi, array_size>());
    types::for_each(simd_test_types(), OveralignedCopyToHelper<T, SimdAbi, array_size>());
  }
};

template <class U, class T, class Flags, class SimdAbi = ex::simd_abi::compatible<T>, class = void>
struct has_copy_from : std::false_type {};

template <class U, class T, class Flags, class SimdAbi>
struct has_copy_from<U,
                     T,
                     Flags,
                     SimdAbi,
                     std::void_t<decltype(std::declval<ex::simd<T, SimdAbi>>().copy_from(
                         std::declval<const U*>(), std::declval<Flags>()))>> : std::true_type {};

template <class U, class T, class Flags, class SimdAbi = ex::simd_abi::compatible<T>, class = void>
struct has_copy_to : std::false_type {};

template <class U, class T, class Flags, class SimdAbi>
struct has_copy_to<
    U,
    T,
    Flags,
    SimdAbi,
    std::void_t<decltype(std::declval<ex::simd<T, SimdAbi>>().copy_to(std::declval<U*>(), std::declval<Flags>()))>>
    : std::true_type {};

template <class T, std::size_t>
struct CheckSimdCopyTraits {
  template <class SimdAbi>
  void operator()() {
    // These functions shall not participate in overload resolution unless
    // is_simd_flag_type_v<Flags> is true, and
    // U is a vectorizable type.
    static_assert(has_copy_from<int, T, ex::element_aligned_tag, SimdAbi>::value);
    static_assert(has_copy_to<int, T, ex::element_aligned_tag, SimdAbi>::value);

    // is_simd_flag_type_v<Flags> is false
    static_assert(!has_copy_from<int, T, T, SimdAbi>::value);
    static_assert(!has_copy_to<int, T, T, SimdAbi>::value);
    static_assert(!has_copy_from<int, T, SimdAbi, SimdAbi>::value);
    static_assert(!has_copy_to<int, T, SimdAbi, SimdAbi>::value);

    // U is not a vectorizable type.
    static_assert(!has_copy_from<SimdAbi, T, ex::element_aligned_tag, SimdAbi>::value);
    static_assert(!has_copy_to<SimdAbi, T, ex::element_aligned_tag, SimdAbi>::value);
    static_assert(!has_copy_from<ex::element_aligned_tag, T, ex::element_aligned_tag, SimdAbi>::value);
    static_assert(!has_copy_to<ex::element_aligned_tag, T, ex::element_aligned_tag, SimdAbi>::value);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdCopyFrom>();
  test_all_simd_abi<CheckSimdCopyTo>();
  test_all_simd_abi<CheckSimdCopyTraits>();
  return 0;
}
