//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Older versions of clang may encounter a backend error (see 0295c2ad):
//   Pass-by-value arguments with alignment greater than register width are not supported.
// XFAIL: target=powerpc{{.*}}-ibm-{{.*}} && (clang-17 || clang-18)

// FIXME: The following issue occurs on macOS 13.7 with Apple clang 15.0.0:
//   clang: error: unable to execute command: Illegal instruction: 4
// XFAIL: target=x86_64-apple-macosx13.7

// <experimental/simd>
//
// [simd.class]
// simd& operator++() noexcept;
// simd operator++(int) noexcept;
// simd& operator--() noexcept;
// simd operator--(int) noexcept;
// mask_type operator!() const noexcept;
// simd operator~() const noexcept;
// simd operator+() const noexcept;
// simd operator-() const noexcept;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdPrefixIncrementOperator {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    static_assert(noexcept(++origin_simd));
    std::array<T, array_size> expected_return_value, expected_value;
    for (size_t i = 0; i < array_size; ++i) {
      expected_return_value[i] = static_cast<T>(i) + 1;
      expected_value[i]        = static_cast<T>(i) + 1;
    }
    assert_simd_values_equal<array_size>(++origin_simd, expected_return_value);
    assert_simd_values_equal<array_size>(origin_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdPostfixIncrementOperator {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    static_assert(noexcept(origin_simd++));
    std::array<T, array_size> expected_return_value, expected_value;
    for (size_t i = 0; i < array_size; ++i) {
      expected_return_value[i] = static_cast<T>(i);
      expected_value[i]        = static_cast<T>(i) + 1;
    }
    assert_simd_values_equal<array_size>(origin_simd++, expected_return_value);
    assert_simd_values_equal<array_size>(origin_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdPrefixDecrementOperator {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    static_assert(noexcept(--origin_simd));
    std::array<T, array_size> expected_return_value, expected_value;
    for (size_t i = 0; i < array_size; ++i) {
      expected_return_value[i] = static_cast<T>(i) - 1;
      expected_value[i]        = static_cast<T>(i) - 1;
    }
    assert_simd_values_equal<array_size>(--origin_simd, expected_return_value);
    assert_simd_values_equal<array_size>(origin_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdPostfixDecrementOperator {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    static_assert(noexcept(origin_simd--));
    std::array<T, array_size> expected_return_value, expected_value;
    for (size_t i = 0; i < array_size; ++i) {
      expected_return_value[i] = static_cast<T>(i);
      expected_value[i]        = static_cast<T>(i) - 1;
    }
    assert_simd_values_equal<array_size>(origin_simd--, expected_return_value);
    assert_simd_values_equal<array_size>(origin_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdNegationOperator {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    static_assert(noexcept(!origin_simd));
    std::array<bool, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = !static_cast<bool>(i);
    assert_simd_mask_values_equal<array_size>(!origin_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBitwiseNotOperator {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    static_assert(noexcept(~origin_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = ~static_cast<T>(i);
    assert_simd_values_equal<array_size>(~origin_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdPositiveSignOperator {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    static_assert(noexcept(+origin_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = +static_cast<T>(i);
    assert_simd_values_equal<array_size>(+origin_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdNegativeSignOperator {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    static_assert(noexcept(-origin_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = -static_cast<T>(i);
    assert_simd_values_equal<array_size>(-origin_simd, expected_value);
  }
};

template <class T, class SimdAbi = ex::simd_abi::compatible<T>, class = void>
struct has_bitwise_not_op : std::false_type {};

template <class T, class SimdAbi>
struct has_bitwise_not_op<T, SimdAbi, std::void_t<decltype(~std::declval<ex::simd<T, SimdAbi>>())>> : std::true_type {};

template <class T, std::size_t>
struct CheckSimdBitwiseNotTraits {
  template <class SimdAbi>
  void operator()() {
    // This function shall not participate in overload resolution unless
    // T is an integral type.
    if constexpr (std::is_integral_v<T>)
      static_assert(has_bitwise_not_op<T, SimdAbi>::value);
    // T is not an integral type.
    else
      static_assert(!has_bitwise_not_op<T, SimdAbi>::value);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdPrefixIncrementOperator>();
  test_all_simd_abi<CheckSimdPostfixIncrementOperator>();
  test_all_simd_abi<CheckSimdPrefixDecrementOperator>();
  test_all_simd_abi<CheckSimdPostfixDecrementOperator>();
  test_all_simd_abi<CheckSimdNegationOperator>();
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckSimdBitwiseNotOperator>());
  test_all_simd_abi<CheckSimdPositiveSignOperator>();
  test_all_simd_abi<CheckSimdNegativeSignOperator>();
  test_all_simd_abi<CheckSimdBitwiseNotTraits>();
  return 0;
}
