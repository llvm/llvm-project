//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Compile-time checks of sycl::half shared by the host and the device code
// tests.

#ifndef LIBSYCL_TEST_BASIC_INPUTS_HALF_COMMON_HPP
#define LIBSYCL_TEST_BASIC_INPUTS_HALF_COMMON_HPP

#include <sycl/sycl.hpp>

#include <limits>
#include <type_traits>
#include <utility>

namespace half_test {

enum UnscopedEnum { EnumValue = 3 };

struct ConvertsToInt {
  constexpr operator int() const { return 2; }
};

// Checks the result types of the binary operators of half and T, in both
// operand orders.
template <typename T, typename ResultT> constexpr void checkOperatorTypes() {
  using std::declval;
  using std::is_same_v;
  using sycl::half;

  static_assert(is_same_v<decltype(declval<half>() + declval<T>()), ResultT>);
  static_assert(is_same_v<decltype(declval<half>() - declval<T>()), ResultT>);
  static_assert(is_same_v<decltype(declval<half>() * declval<T>()), ResultT>);
  static_assert(is_same_v<decltype(declval<half>() / declval<T>()), ResultT>);
  static_assert(is_same_v<decltype(declval<T>() + declval<half>()), ResultT>);
  static_assert(is_same_v<decltype(declval<T>() - declval<half>()), ResultT>);
  static_assert(is_same_v<decltype(declval<T>() * declval<half>()), ResultT>);
  static_assert(is_same_v<decltype(declval<T>() / declval<half>()), ResultT>);

  static_assert(is_same_v<decltype(declval<half>() == declval<T>()), bool>);
  static_assert(is_same_v<decltype(declval<half>() != declval<T>()), bool>);
  static_assert(is_same_v<decltype(declval<half>() < declval<T>()), bool>);
  static_assert(is_same_v<decltype(declval<half>() > declval<T>()), bool>);
  static_assert(is_same_v<decltype(declval<half>() <= declval<T>()), bool>);
  static_assert(is_same_v<decltype(declval<half>() >= declval<T>()), bool>);
  static_assert(is_same_v<decltype(declval<T>() == declval<half>()), bool>);
  static_assert(is_same_v<decltype(declval<T>() != declval<half>()), bool>);
  static_assert(is_same_v<decltype(declval<T>() < declval<half>()), bool>);
  static_assert(is_same_v<decltype(declval<T>() > declval<half>()), bool>);
  static_assert(is_same_v<decltype(declval<T>() <= declval<half>()), bool>);
  static_assert(is_same_v<decltype(declval<T>() >= declval<half>()), bool>);

  static_assert(is_same_v<decltype(declval<half &>() += declval<T>()), half &>);
  static_assert(is_same_v<decltype(declval<half &>() -= declval<T>()), half &>);
  static_assert(is_same_v<decltype(declval<half &>() *= declval<T>()), half &>);
  static_assert(is_same_v<decltype(declval<half &>() /= declval<T>()), half &>);
}

// Checks the operator result types for every arithmetic operand type, and for
// operands that convert implicitly to one: the result type follows the usual
// arithmetic conversions, with half taking the place of a floating-point type
// narrower than float.
constexpr void checkAllOperatorTypes() {
  using std::declval;
  using std::is_same_v;
  using sycl::half;

  checkOperatorTypes<half, half>();
  checkOperatorTypes<float, float>();
  checkOperatorTypes<double, double>();
  checkOperatorTypes<long double, long double>();

  checkOperatorTypes<bool, half>();
  checkOperatorTypes<char, half>();
  checkOperatorTypes<signed char, half>();
  checkOperatorTypes<unsigned char, half>();
  checkOperatorTypes<short, half>();
  checkOperatorTypes<unsigned short, half>();
  checkOperatorTypes<int, half>();
  checkOperatorTypes<unsigned int, half>();
  checkOperatorTypes<long, half>();
  checkOperatorTypes<unsigned long, half>();
  checkOperatorTypes<long long, half>();
  checkOperatorTypes<unsigned long long, half>();

  checkOperatorTypes<UnscopedEnum, half>();
  checkOperatorTypes<ConvertsToInt, half>();
  checkOperatorTypes<sycl::id<1>, half>();

  static_assert(is_same_v<decltype(-declval<half>()), half>);
  static_assert(is_same_v<decltype(+declval<half>()), half>);
  static_assert(is_same_v<decltype(++declval<half &>()), half &>);
  static_assert(is_same_v<decltype(--declval<half &>()), half &>);
  static_assert(is_same_v<decltype(declval<half &>()++), half>);
  static_assert(is_same_v<decltype(declval<half &>()--), half>);
}

// Checks that every std::numeric_limits<half> function is usable in constant
// expressions.
inline void checkConstexprLimits() {
  using Limits = std::numeric_limits<sycl::half>;
  constexpr sycl::half Values[] = {
      (Limits::min)(),     (Limits::max)(),       Limits::lowest(),
      Limits::epsilon(),   Limits::round_error(), Limits::infinity(),
      -Limits::infinity(), Limits::quiet_NaN(),   Limits::signaling_NaN(),
      Limits::denorm_min()};
  static_cast<void>(Values);
}

} // namespace half_test

#endif // LIBSYCL_TEST_BASIC_INPUTS_HALF_COMMON_HPP
