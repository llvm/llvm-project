//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <utility>

// template<class T, class U>
//   constexpr bool cmp_equal(T t, U u) noexcept; // C++20

// template<class T, class U>
//   constexpr bool cmp_not_equal(T t, U u) noexcept; // C++20

// template<class T, class U>
//   constexpr bool cmp_less(T t, U u) noexcept; // C++20

// template<class T, class U>
//   constexpr bool cmp_less_equal(T t, U u) noexcept; // C++20

// template<class T, class U>
//   constexpr bool cmp_greater(T t, U u) noexcept; // C++20

// template<class T, class U>
//   constexpr bool cmp_greater_equal(T t, U u) noexcept; // C++20

// template<class R, class T>
//   constexpr bool in_range(T t) noexcept;      // C++20

#include <utility>
#include <cstddef>

#include "test_macros.h"

struct NonEmptyT {
  int val;
  NonEmptyT() : val(0) {}
  NonEmptyT(int val) : val(val) {}
  operator int&() { return val; }
  operator int() const { return val; }
};

enum ColorT { red, green, blue };

struct EmptyT {};

template <class T>
constexpr void test() {
  std::cmp_equal(T(), T()); // expected-error11{{no matching function for call to 'cmp_equal'}}
  std::cmp_equal(T(), int()); // expected-error11{{no matching function for call to 'cmp_equal'}}
  std::cmp_equal(int(), T()); // expected-error11{{no matching function for call to 'cmp_equal'}}
  std::cmp_not_equal(T(), T()); // expected-error11{{no matching function for call to 'cmp_not_equal'}}
  std::cmp_not_equal(T(), int()); // expected-error11{{no matching function for call to 'cmp_not_equal'}}
  std::cmp_not_equal(int(), T()); // expected-error11{{no matching function for call to 'cmp_not_equal'}}
  std::cmp_less(T(), T()); // expected-error11{{no matching function for call to 'cmp_less'}}
  std::cmp_less(T(), int()); // expected-error11{{no matching function for call to 'cmp_less'}}
  std::cmp_less(int(), T()); // expected-error11{{no matching function for call to 'cmp_less'}}
  std::cmp_less_equal(T(), T()); // expected-error11{{no matching function for call to 'cmp_less_equal'}}
  std::cmp_less_equal(T(), int()); // expected-error11{{no matching function for call to 'cmp_less_equal'}}
  std::cmp_less_equal(int(), T()); // expected-error11{{no matching function for call to 'cmp_less_equal'}}
  std::cmp_greater(T(), T()); // expected-error11{{no matching function for call to 'cmp_greater'}}
  std::cmp_greater(T(), int()); // expected-error11{{no matching function for call to 'cmp_greater'}}
  std::cmp_greater(int(), T()); // expected-error11{{no matching function for call to 'cmp_greater'}}
  std::cmp_greater_equal(T(), T()); // expected-error11{{no matching function for call to 'cmp_greater_equal'}}
  std::cmp_greater_equal(T(), int()); // expected-error11{{no matching function for call to 'cmp_greater_equal'}}
  std::cmp_greater_equal(int(), T()); // expected-error11{{no matching function for call to 'cmp_greater_equal'}}
  std::in_range<T>(int()); // expected-error11{{no matching function for call to 'in_range'}}
  std::in_range<int>(T()); // expected-error11{{no matching function for call to 'in_range'}}
}
#ifndef _LIBCPP_HAS_NO_CHAR8_T
template <class T>
constexpr void test_char8t() {
  std::cmp_equal(T(), T()); // expected-error1{{no matching function for call to 'cmp_equal'}}
  std::cmp_equal(T(), int()); // expected-error1{{no matching function for call to 'cmp_equal'}}
  std::cmp_equal(int(), T()); // expected-error1{{no matching function for call to 'cmp_equal'}}
  std::cmp_not_equal(T(), T()); // expected-error1{{no matching function for call to 'cmp_not_equal'}}
  std::cmp_not_equal(T(), int()); // expected-error1{{no matching function for call to 'cmp_not_equal'}}
  std::cmp_not_equal(int(), T()); // expected-error1{{no matching function for call to 'cmp_not_equal'}}
  std::cmp_less(T(), T()); // expected-error1{{no matching function for call to 'cmp_less'}}
  std::cmp_less(T(), int()); // expected-error1{{no matching function for call to 'cmp_less'}}
  std::cmp_less(int(), T()); // expected-error1{{no matching function for call to 'cmp_less'}}
  std::cmp_less_equal(T(), T()); // expected-error1{{no matching function for call to 'cmp_less_equal'}}
  std::cmp_less_equal(T(), int()); // expected-error1{{no matching function for call to 'cmp_less_equal'}}
  std::cmp_less_equal(int(), T()); // expected-error1{{no matching function for call to 'cmp_less_equal'}}
  std::cmp_greater(T(), T()); // expected-error1{{no matching function for call to 'cmp_greater'}}
  std::cmp_greater(T(), int()); // expected-error1{{no matching function for call to 'cmp_greater'}}
  std::cmp_greater(int(), T()); // expected-error1{{no matching function for call to 'cmp_greater'}}
  std::cmp_greater_equal(T(), T()); // expected-error1{{no matching function for call to 'cmp_greater_equal'}}
  std::cmp_greater_equal(T(), int()); // expected-error1{{no matching function for call to 'cmp_greater_equal'}}
  std::cmp_greater_equal(int(), T()); // expected-error1{{no matching function for call to 'cmp_greater_equal'}}
  std::in_range<T>(int()); // expected-error1{{no matching function for call to 'in_range'}}
  std::in_range<int>(T()); // expected-error1{{no matching function for call to 'in_range'}}
}
#endif // _LIBCPP_HAS_NO_CHAR8_T

#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
template <class T>
constexpr void test_uchars() {
  std::cmp_equal(T(), T()); // expected-error2{{no matching function for call to 'cmp_equal'}}
  std::cmp_equal(T(), int()); // expected-error2{{no matching function for call to 'cmp_equal'}}
  std::cmp_equal(int(), T()); // expected-error2{{no matching function for call to 'cmp_equal'}}
  std::cmp_not_equal(T(), T()); // expected-error2{{no matching function for call to 'cmp_not_equal'}}
  std::cmp_not_equal(T(), int()); // expected-error2{{no matching function for call to 'cmp_not_equal'}}
  std::cmp_not_equal(int(), T()); // expected-error2{{no matching function for call to 'cmp_not_equal'}}
  std::cmp_less(T(), T()); // expected-error2{{no matching function for call to 'cmp_less'}}
  std::cmp_less(T(), int()); // expected-error2{{no matching function for call to 'cmp_less'}}
  std::cmp_less(int(), T()); // expected-error2{{no matching function for call to 'cmp_less'}}
  std::cmp_less_equal(T(), T()); // expected-error2{{no matching function for call to 'cmp_less_equal'}}
  std::cmp_less_equal(T(), int()); // expected-error2{{no matching function for call to 'cmp_less_equal'}}
  std::cmp_less_equal(int(), T()); // expected-error2{{no matching function for call to 'cmp_less_equal'}}
  std::cmp_greater(T(), T()); // expected-error2{{no matching function for call to 'cmp_greater'}}
  std::cmp_greater(T(), int()); // expected-error2{{no matching function for call to 'cmp_greater'}}
  std::cmp_greater(int(), T()); // expected-error2{{no matching function for call to 'cmp_greater'}}
  std::cmp_greater_equal(T(), T()); // expected-error2{{no matching function for call to 'cmp_greater_equal'}}
  std::cmp_greater_equal(T(), int()); // expected-error2{{no matching function for call to 'cmp_greater_equal'}}
  std::cmp_greater_equal(int(), T()); // expected-error2{{no matching function for call to 'cmp_greater_equal'}}
  std::in_range<T>(int()); // expected-error2{{no matching function for call to 'in_range'}}
  std::in_range<int>(T()); // expected-error2{{no matching function for call to 'in_range'}}
}
#endif // _LIBCPP_HAS_NO_UNICODE_CHARS

int main() {
  test<bool>();
  test<char>();
  test<wchar_t>();
  test<float>();
  test<double>();
  test<long double>();
  test<std::byte>();
  test<NonEmptyT>();
  test<ColorT>();
  test<std::nullptr_t>();
  test<EmptyT>();

#ifndef _LIBCPP_HAS_NO_CHAR8_T
  test_char8t<char8_t>();
#endif // _LIBCPP_HAS_NO_CHAR8_T

#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
  test_uchars<char16_t>();
  test_uchars<char32_t>();
#endif

  return 0;
}
