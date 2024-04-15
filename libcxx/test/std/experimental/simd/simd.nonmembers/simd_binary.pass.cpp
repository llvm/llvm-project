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
// [simd.nonmembers]
// friend simd operator+(const simd& lhs, const simd& rhs) noexcept;
// friend simd operator-(const simd& lhs, const simd& rhs) noexcept;
// friend simd operator*(const simd& lhs, const simd& rhs) noexcept;
// friend simd operator/(const simd& lhs, const simd& rhs) noexcept;
// friend simd operator%(const simd& lhs, const simd& rhs) noexcept;
// friend simd operator&(const simd& lhs, const simd& rhs) noexcept;
// friend simd operator|(const simd& lhs, const simd& rhs) noexcept;
// friend simd operator^(const simd& lhs, const simd& rhs) noexcept;
// friend simd operator<<(const simd& lhs, const simd& rhs) noexcept;
// friend simd operator>>(const simd& lhs, const simd& rhs) noexcept;
// friend simd operator<<(const simd& v, int n) noexcept;
// friend simd operator>>(const simd& v, int n) noexcept;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdBinaryOperatorPlus {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> left_simd([](T i) { return i; });
    ex::simd<T, SimdAbi> right_simd(static_cast<T>(2));
    static_assert(noexcept(left_simd + right_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) + static_cast<T>(2);
    assert_simd_values_equal<array_size>(left_simd + right_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBinaryOperatorMinus {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> left_simd([](T i) { return i; });
    ex::simd<T, SimdAbi> right_simd(static_cast<T>(2));
    static_assert(noexcept(left_simd - right_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) - static_cast<T>(2);
    assert_simd_values_equal<array_size>(left_simd - right_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBinaryOperatorMultiplies {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> left_simd([](T i) { return i; });
    ex::simd<T, SimdAbi> right_simd(static_cast<T>(2));
    static_assert(noexcept(left_simd * right_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) * static_cast<T>(2);
    assert_simd_values_equal<array_size>(left_simd * right_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBinaryOperatorDivides {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> left_simd([](T i) { return i; });
    ex::simd<T, SimdAbi> right_simd(static_cast<T>(2));
    static_assert(noexcept(left_simd / right_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) / static_cast<T>(2);
    assert_simd_values_equal<array_size>(left_simd / right_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBinaryOperatorModulus {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> left_simd([](T i) { return i; });
    ex::simd<T, SimdAbi> right_simd(static_cast<T>(2));
    static_assert(noexcept(left_simd % right_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) % static_cast<T>(2);
    assert_simd_values_equal<array_size>(left_simd % right_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBinaryOperatorBitAnd {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> left_simd([](T i) { return i; });
    ex::simd<T, SimdAbi> right_simd(static_cast<T>(2));
    static_assert(noexcept(left_simd & right_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) & static_cast<T>(2);
    assert_simd_values_equal<array_size>(left_simd & right_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBinaryOperatorBitOr {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> left_simd([](T i) { return i; });
    ex::simd<T, SimdAbi> right_simd(static_cast<T>(2));
    static_assert(noexcept(left_simd | right_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) | static_cast<T>(2);
    assert_simd_values_equal<array_size>(left_simd | right_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBinaryOperatorBitXor {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> left_simd([](T i) { return i; });
    ex::simd<T, SimdAbi> right_simd(static_cast<T>(2));
    static_assert(noexcept(left_simd ^ right_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) ^ static_cast<T>(2);
    assert_simd_values_equal<array_size>(left_simd ^ right_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBinaryOperatorShiftLeft {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> left_simd([](T i) { return i; });
    ex::simd<T, SimdAbi> right_simd(static_cast<T>(2));
    static_assert(noexcept(left_simd << right_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) << static_cast<T>(2);
    assert_simd_values_equal<array_size>(left_simd << right_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBinaryOperatorShiftRight {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> left_simd([](T i) { return i; });
    ex::simd<T, SimdAbi> right_simd(static_cast<T>(2));
    static_assert(noexcept(left_simd >> right_simd));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) >> static_cast<T>(2);
    assert_simd_values_equal<array_size>(left_simd >> right_simd, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBinaryOperatorShiftLeftByInt {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> simd_value([](T i) { return i; });
    constexpr int shift_amount = 2;
    static_assert(noexcept(simd_value << shift_amount));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) << shift_amount;
    assert_simd_values_equal<array_size>(simd_value << shift_amount, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdBinaryOperatorShiftRightByInt {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    ex::simd<T, SimdAbi> simd_value([](T i) { return i; });
    constexpr int shift_amount = 2;
    static_assert(noexcept(simd_value >> shift_amount));
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(i) >> shift_amount;
    assert_simd_values_equal<array_size>(simd_value >> shift_amount, expected_value);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdBinaryOperatorPlus>();
  test_all_simd_abi<CheckSimdBinaryOperatorMinus>();
  test_all_simd_abi<CheckSimdBinaryOperatorMultiplies>();
  test_all_simd_abi<CheckSimdBinaryOperatorDivides>();
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckSimdBinaryOperatorModulus>());
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckSimdBinaryOperatorBitAnd>());
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckSimdBinaryOperatorBitOr>());
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckSimdBinaryOperatorBitXor>());
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckSimdBinaryOperatorShiftLeft>());
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckSimdBinaryOperatorShiftRight>());
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckSimdBinaryOperatorShiftLeftByInt>());
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckSimdBinaryOperatorShiftRightByInt>());
  return 0;
}
