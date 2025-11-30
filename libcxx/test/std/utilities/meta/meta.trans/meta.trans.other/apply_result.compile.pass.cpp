//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <type_traits>

// template<class Fn, class Tuple> struct apply_result;

// template<class Fn, class Tuple>
// using apply_result_t = typename apply_result<T, U>::type;

#include <cassert>
#include <cstddef>
#include <array>
#include <complex>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

#include "callable_types.h"
#include "test_iterators.h"

struct empty_aggregate {};

struct derived_from_tuple_int : std::tuple<int> {};

template <>
struct std::tuple_size<derived_from_tuple_int> : std::integral_constant<std::size_t, 1> {};

template <std::size_t I>
  requires(I < 1)
struct std::tuple_element<I, derived_from_tuple_int> {
  using type = std::tuple_element_t<I, std::tuple<int>>;
};

template <class Fn, class Tuple>
concept apply_result_has_member_type = requires { typename std::apply_result<Fn, Tuple>::type; };

template <class Fn, class Tuple>
concept apply_result_t_is_valid = requires { typename std::apply_result_t<Fn, Tuple>; };

template <class Fn, class Tuple, class Expected>
void test_valid_apply_result() {
  static_assert(apply_result_has_member_type<Fn, Tuple>);
  static_assert(apply_result_t_is_valid<Fn, Tuple>);
  static_assert(std::is_same_v<typename std::apply_result<Fn, Tuple>::type, Expected>);
  static_assert(std::is_same_v<std::apply_result_t<Fn, Tuple>, Expected>);
}

template <class Func, class Tuple, class Expected>
void test_valid_apply_result_from_function() {
  static_assert(std::is_function_v<Func>);

  test_valid_apply_result<Func, Tuple, Expected>();
  test_valid_apply_result<Func&, Tuple, Expected>();

  test_valid_apply_result<Func*, Tuple, Expected>();
  test_valid_apply_result<Func*&, Tuple, Expected>();
  test_valid_apply_result<Func* const, Tuple, Expected>();
  test_valid_apply_result<Func* const&, Tuple, Expected>();
  test_valid_apply_result<Func* volatile, Tuple, Expected>();
  test_valid_apply_result<Func* volatile&, Tuple, Expected>();
  test_valid_apply_result<Func* const volatile, Tuple, Expected>();
  test_valid_apply_result<Func* const volatile&, Tuple, Expected>();
}

template <class Fn, class Tuple>
void test_invalid_apply_result() {
  static_assert(!apply_result_has_member_type<Fn, Tuple>);
  static_assert(!apply_result_t_is_valid<Fn, Tuple>);
}

template <class Func, class Tuple>
void test_invalid_apply_result_from_function() {
  static_assert(std::is_function_v<Func>);

  test_invalid_apply_result<Func, Tuple>();
  test_invalid_apply_result<Func&, Tuple>();

  test_invalid_apply_result<Func*, Tuple>();
  test_invalid_apply_result<Func*&, Tuple>();
  test_invalid_apply_result<Func* const, Tuple>();
  test_invalid_apply_result<Func* const&, Tuple>();
  test_invalid_apply_result<Func* volatile, Tuple>();
  test_invalid_apply_result<Func* volatile&, Tuple>();
  test_invalid_apply_result<Func* const volatile, Tuple>();
  test_invalid_apply_result<Func* const volatile&, Tuple>();
}

void test_valid() {
  // test array
  test_valid_apply_result_from_function<int(), std::array<int, 0>, int>();
  test_valid_apply_result_from_function<int(), std::array<long, 0>&, int>();
  test_valid_apply_result_from_function<int(), const std::array<char, 0>, int>();
  test_valid_apply_result_from_function<int(), const std::array<std::array<int, 1>, 0>&, int>();
  test_valid_apply_result_from_function<int() noexcept, std::array<int, 0>, int>();
  test_valid_apply_result_from_function<int() noexcept, std::array<long, 0>&, int>();
  test_valid_apply_result_from_function<int() noexcept, const std::array<char, 0>, int>();
  test_valid_apply_result_from_function<int() noexcept, const std::array<std::array<int, 1>, 0>&, int>();

  test_valid_apply_result_from_function<int(long), std::array<int, 1>, int>();
  test_valid_apply_result_from_function<int&(int), std::array<long, 1>&, int&>();
  test_valid_apply_result_from_function<const int&&(float), const std::array<double, 1>, const int&&>();
  test_valid_apply_result_from_function<void(double), const std::array<char, 1>&, void>();
  test_valid_apply_result_from_function<int(long) noexcept, std::array<int, 1>, int>();
  test_valid_apply_result_from_function<int&(int) noexcept, std::array<long, 1>&, int&>();
  test_valid_apply_result_from_function<const int&&(float) noexcept, const std::array<double, 1>, const int&&>();
  test_valid_apply_result_from_function<void(double) noexcept, const std::array<char, 1>&, void>();

  test_valid_apply_result_from_function<int(long, int), std::array<int, 2>, int>();
  test_valid_apply_result_from_function<int&(long, int), std::array<int, 2>&, int&>();
  test_valid_apply_result_from_function<const int&&(long, int), const std::array<int, 2>, const int&&>();
  test_valid_apply_result_from_function<void(long, int), const std::array<int, 2>&, void>();
  test_valid_apply_result_from_function<int(long, int) noexcept, std::array<int, 2>, int>();
  test_valid_apply_result_from_function<int&(long, int) noexcept, std::array<int, 2>&, int&>();
  test_valid_apply_result_from_function<const int&&(long, int) noexcept, const std::array<int, 2>, const int&&>();
  test_valid_apply_result_from_function<void(long, int) noexcept, const std::array<int, 2>&, void>();

  test_valid_apply_result<ConstCallable<bool>, std::array<int, 0>, bool>();
  test_valid_apply_result<ConstCallable<signed char>, std::array<int, 1>&, signed char>();
  test_valid_apply_result<ConstCallable<short>, const std::array<int, 2>, short>();
  test_valid_apply_result<ConstCallable<int>, const std::array<int, 3>&, int>();

  test_valid_apply_result<NoExceptCallable<bool>, std::array<int, 0>, bool>();
  test_valid_apply_result<NoExceptCallable<unsigned char>, std::array<int, 1>&, unsigned char>();
  test_valid_apply_result<NoExceptCallable<unsigned short>, const std::array<int, 2>, unsigned short>();
  test_valid_apply_result<NoExceptCallable<unsigned int>, const std::array<int, 3>&, unsigned int>();

  // test complex
  test_valid_apply_result_from_function<float(float, float), std::complex<float>, float>();
  test_valid_apply_result_from_function<float(float&, float&), std::complex<float>&, float>();
  test_valid_apply_result_from_function<void(float, float), const std::complex<float>, void>();
  test_valid_apply_result_from_function<double(float, float), const std::complex<float>&, double>();
  test_valid_apply_result_from_function<float(float, float) noexcept, std::complex<float>, float>();
  test_valid_apply_result_from_function<float(float&, float&) noexcept, std::complex<float>&, float>();
  test_valid_apply_result_from_function<void(float, float) noexcept, const std::complex<float>, void>();
  test_valid_apply_result_from_function<double(float, float) noexcept, const std::complex<float>&, double>();

  test_valid_apply_result<ConstCallable<bool>, std::complex<float>, bool>();
  test_valid_apply_result<ConstCallable<signed char>, std::complex<float>&, signed char>();
  test_valid_apply_result<ConstCallable<int>, const std::complex<float>, int>();
  test_valid_apply_result<ConstCallable<short>, const std::complex<float>&, short>();

  test_valid_apply_result<NoExceptCallable<unsigned long>, std::complex<float>, unsigned long>();
  test_valid_apply_result<NoExceptCallable<bool>, std::complex<float>&, bool>();
  test_valid_apply_result<NoExceptCallable<unsigned int>, const std::complex<float>, unsigned int>();
  test_valid_apply_result<NoExceptCallable<unsigned short>, const std::complex<float>&, unsigned short>();

  test_valid_apply_result_from_function<double(double, double), std::complex<double>, double>();
  test_valid_apply_result_from_function<double&(double&, double&), std::complex<double>&, double&>();
  test_valid_apply_result_from_function<void(double, double), const std::complex<double>, void>();
  test_valid_apply_result_from_function<double(double, double), const std::complex<double>&, double>();
  test_valid_apply_result_from_function<double(double, double) noexcept, std::complex<double>, double>();
  test_valid_apply_result_from_function<double&(double&, double&) noexcept, std::complex<double>&, double&>();
  test_valid_apply_result_from_function<void(double, double) noexcept, const std::complex<double>, void>();
  test_valid_apply_result_from_function<double(double, double) noexcept, const std::complex<double>&, double>();

  test_valid_apply_result<ConstCallable<bool>, std::complex<double>, bool>();
  test_valid_apply_result<ConstCallable<signed char>, std::complex<double>&, signed char>();
  test_valid_apply_result<ConstCallable<short>, const std::complex<double>, short>();
  test_valid_apply_result<ConstCallable<int>, const std::complex<double>&, int>();

  test_valid_apply_result<NoExceptCallable<unsigned short>, std::complex<double>, unsigned short>();
  test_valid_apply_result<NoExceptCallable<unsigned long>, std::complex<double>&, unsigned long>();
  test_valid_apply_result<NoExceptCallable<bool>, const std::complex<double>, bool>();
  test_valid_apply_result<NoExceptCallable<unsigned char>, const std::complex<double>&, unsigned char>();

  test_valid_apply_result_from_function<long double(long double, long double),
                                        std::complex<long double>,
                                        long double>();
  test_valid_apply_result_from_function<long double&(long double&, long double&),
                                        std::complex<long double>&,
                                        long double&>();
  test_valid_apply_result_from_function<void(long double, long double), const std::complex<long double>, void>();
  test_valid_apply_result_from_function<double(long double, long double), const std::complex<long double>&, double>();
  test_valid_apply_result_from_function<long double(long double, long double) noexcept,
                                        std::complex<long double>,
                                        long double>();
  test_valid_apply_result_from_function<long double&(long double&, long double&) noexcept,
                                        std::complex<long double>&,
                                        long double&>();
  test_valid_apply_result_from_function<void(long double, long double) noexcept,
                                        const std::complex<long double>,
                                        void>();
  test_valid_apply_result_from_function<double(long double, long double) noexcept,
                                        const std::complex<long double>&,
                                        double>();

  test_valid_apply_result<ConstCallable<bool>, std::complex<long double>, bool>();
  test_valid_apply_result<ConstCallable<signed char>, std::complex<long double>&, signed char>();
  test_valid_apply_result<ConstCallable<short>, const std::complex<long double>, short>();
  test_valid_apply_result<ConstCallable<int>, const std::complex<long double>&, int>();

  test_valid_apply_result<NoExceptCallable<bool>, std::complex<long double>, bool>();
  test_valid_apply_result<NoExceptCallable<unsigned char>, std::complex<long double>&, unsigned char>();
  test_valid_apply_result<NoExceptCallable<unsigned short>, const std::complex<long double>, unsigned short>();
  test_valid_apply_result<NoExceptCallable<unsigned int>, const std::complex<long double>&, unsigned int>();

  // test subrange
  using copyable_subrange = std::ranges::subrange<int*>;

  test_valid_apply_result_from_function<void(int*, int*), copyable_subrange, void>();
  test_valid_apply_result_from_function<long(int*, int*), copyable_subrange&, long>();
  test_valid_apply_result_from_function<int*&(int*, int*), const copyable_subrange, int*&>();
  test_valid_apply_result_from_function<int* && (int*, int*), const copyable_subrange&, int*&&>();
  test_valid_apply_result_from_function<void(int*, int*) noexcept, copyable_subrange, void>();
  test_valid_apply_result_from_function<long(int*, int*) noexcept, copyable_subrange&, long>();
  test_valid_apply_result_from_function<int*&(int*, int*) noexcept, const copyable_subrange, int*&>();
  test_valid_apply_result_from_function<int* && (int*, int*) noexcept, const copyable_subrange&, int*&&>();

  test_valid_apply_result<ConstCallable<bool>, copyable_subrange, bool>();
  test_valid_apply_result<ConstCallable<unsigned char>, copyable_subrange&, unsigned char>();
  test_valid_apply_result<ConstCallable<short>, const copyable_subrange, short>();
  test_valid_apply_result<ConstCallable<unsigned long>, const copyable_subrange&, unsigned long>();

  test_valid_apply_result<NoExceptCallable<signed char>, copyable_subrange, signed char>();
  test_valid_apply_result<NoExceptCallable<unsigned int>, copyable_subrange&, unsigned int>();
  test_valid_apply_result<NoExceptCallable<long long>, const copyable_subrange, long long>();
  test_valid_apply_result<NoExceptCallable<unsigned short>, const copyable_subrange&, unsigned short>();

  using move_only_counted_iter = std::counted_iterator<cpp20_input_iterator<int*>>;
  using move_only_subrange     = std::ranges::subrange<move_only_counted_iter, std::default_sentinel_t>;

  test_valid_apply_result_from_function<void(move_only_counted_iter&&, std::default_sentinel_t),
                                        move_only_subrange,
                                        void>();
  test_valid_apply_result_from_function<void*(move_only_counted_iter&&, std::default_sentinel_t) noexcept,
                                        move_only_subrange,
                                        void*>();

  test_valid_apply_result<ConstCallable<long long>, move_only_subrange, long long>();

  test_valid_apply_result<NoExceptCallable<unsigned long long>, move_only_subrange, unsigned long long>();

  // test tuple
  test_valid_apply_result_from_function<int(), std::tuple<>, int>();
  test_valid_apply_result_from_function<char&(), std::tuple<>&, char&>();
  test_valid_apply_result_from_function<long&&(), const std::tuple<>, long&&>();
  test_valid_apply_result_from_function<void(), const std::tuple<>&, void>();
  test_valid_apply_result_from_function<int() noexcept, std::tuple<>, int>();
  test_valid_apply_result_from_function<char&() noexcept, std::tuple<>&, char&>();
  test_valid_apply_result_from_function<long&&() noexcept, const std::tuple<>, long&&>();
  test_valid_apply_result_from_function<void() noexcept, const std::tuple<>&, void>();

  test_valid_apply_result_from_function<int(long, int), std::tuple<int, long>, int>();
  test_valid_apply_result_from_function<int&(long, int), std::tuple<int, long>&, int&>();
  test_valid_apply_result_from_function<const int&&(long, int), const std::tuple<int, long>, const int&&>();
  test_valid_apply_result_from_function<void(long, int), const std::tuple<int, long>&, void>();
  test_valid_apply_result_from_function<int(long, int) noexcept, std::tuple<int, long>, int>();
  test_valid_apply_result_from_function<int&(long, int) noexcept, std::tuple<int, long>&, int&>();
  test_valid_apply_result_from_function<const int&&(long, int) noexcept, const std::tuple<int, long>, const int&&>();
  test_valid_apply_result_from_function<void(long, int) noexcept, const std::tuple<int, long>&, void>();

  test_valid_apply_result<ConstCallable<unsigned long long>, std::tuple<>, unsigned long long>();
  test_valid_apply_result<ConstCallable<unsigned long>, std::tuple<long>&, unsigned long>();
  test_valid_apply_result<ConstCallable<unsigned int>, const std::tuple<int, long>, unsigned int>();
  test_valid_apply_result<ConstCallable<unsigned short>, const std::tuple<int, double, long>&, unsigned short>();

  test_valid_apply_result<NoExceptCallable<long long>, std::tuple<>, long long>();
  test_valid_apply_result<NoExceptCallable<long>, std::tuple<long>&, long>();
  test_valid_apply_result<NoExceptCallable<int>, const std::tuple<int, long>, int>();
  test_valid_apply_result<NoExceptCallable<short>, const std::tuple<int, double, long>&, short>();

  // test pair
  test_valid_apply_result_from_function<int(long, int), std::pair<int, long>, int>();
  test_valid_apply_result_from_function<int&(long, int), std::pair<int, long>&, int&>();
  test_valid_apply_result_from_function<const int&&(long, int), const std::pair<int, long>, const int&&>();
  test_valid_apply_result_from_function<void(long, int), const std::pair<int, long>&, void>();
  test_valid_apply_result_from_function<int(long, int) noexcept, std::pair<int, long>, int>();
  test_valid_apply_result_from_function<int&(long, int) noexcept, std::pair<int, long>&, int&>();
  test_valid_apply_result_from_function<const int&&(long, int) noexcept, const std::pair<int, long>, const int&&>();
  test_valid_apply_result_from_function<void(long, int) noexcept, const std::pair<int, long>&, void>();

  test_valid_apply_result<ConstCallable<bool>, std::pair<char, wchar_t>, bool>();
  test_valid_apply_result<ConstCallable<unsigned char>, std::pair<float, long>&, unsigned char>();
  test_valid_apply_result<ConstCallable<unsigned int>, const std::pair<int, long>, unsigned int>();
  test_valid_apply_result<ConstCallable<unsigned short>, const std::pair<int, double>&, unsigned short>();

  test_valid_apply_result<NoExceptCallable<int>, std::pair<char, wchar_t>, int>();
  test_valid_apply_result<NoExceptCallable<short>, std::pair<float, long>&, short>();
  test_valid_apply_result<NoExceptCallable<long>, const std::pair<int, long>, long>();
  test_valid_apply_result<NoExceptCallable<long long>, const std::pair<int, double>&, long long>();
}

void test_volatile() {
  // test array
  test_invalid_apply_result_from_function<void(int), volatile std::array<int, 1>>();
  test_invalid_apply_result_from_function<void(int), volatile std::array<int, 1>&>();
  test_invalid_apply_result_from_function<void(int), const volatile std::array<int, 1>>();
  test_invalid_apply_result_from_function<void(int), const volatile std::array<int, 1>&>();
  test_invalid_apply_result_from_function<void(int) noexcept, volatile std::array<int, 1>>();
  test_invalid_apply_result_from_function<void(int) noexcept, volatile std::array<int, 1>&>();
  test_invalid_apply_result_from_function<void(int) noexcept, const volatile std::array<int, 1>>();
  test_invalid_apply_result_from_function<void(int) noexcept, const volatile std::array<int, 1>&>();

  test_invalid_apply_result<NoExceptCallable<bool>, volatile std::array<int, 1>>();
  test_invalid_apply_result<NoExceptCallable<bool>, volatile std::array<int, 1>&>();
  test_invalid_apply_result<NoExceptCallable<bool>, const volatile std::array<int, 1>>();
  test_invalid_apply_result<NoExceptCallable<bool>, const volatile std::array<int, 1>&>();

  // test complex
  test_invalid_apply_result_from_function<void(double, double), volatile std::complex<double>>();
  test_invalid_apply_result_from_function<void(double, double), volatile std::complex<double>&>();
  test_invalid_apply_result_from_function<void(double, double), const volatile std::complex<double>>();
  test_invalid_apply_result_from_function<void(double, double), const volatile std::complex<double>&>();
  test_invalid_apply_result_from_function<void(double, double) noexcept, volatile std::complex<double>>();
  test_invalid_apply_result_from_function<void(double, double) noexcept, volatile std::complex<double>&>();
  test_invalid_apply_result_from_function<void(double, double) noexcept, const volatile std::complex<double>>();
  test_invalid_apply_result_from_function<void(double, double) noexcept, const volatile std::complex<double>&>();

  test_invalid_apply_result<NoExceptCallable<bool>, volatile std::complex<double>>();
  test_invalid_apply_result<NoExceptCallable<bool>, volatile std::complex<double>&>();
  test_invalid_apply_result<NoExceptCallable<bool>, const volatile std::complex<double>>();
  test_invalid_apply_result<NoExceptCallable<bool>, const volatile std::complex<double>&>();

  // test subrange
  using copyable_subrange = std::ranges::subrange<int*>;

  test_invalid_apply_result_from_function<void(int*, int*), volatile copyable_subrange>();
  test_invalid_apply_result_from_function<void(int*, int*), volatile copyable_subrange&>();
  test_invalid_apply_result_from_function<void(int*, int*), const volatile copyable_subrange>();
  test_invalid_apply_result_from_function<void(int*, int*), const volatile copyable_subrange&>();
  test_invalid_apply_result_from_function<void(int*, int*) noexcept, volatile copyable_subrange>();
  test_invalid_apply_result_from_function<void(int*, int*) noexcept, volatile copyable_subrange&>();
  test_invalid_apply_result_from_function<void(int*, int*) noexcept, const volatile copyable_subrange>();
  test_invalid_apply_result_from_function<void(int*, int*) noexcept, const volatile copyable_subrange&>();

  test_invalid_apply_result<NoExceptCallable<bool>, volatile copyable_subrange>();
  test_invalid_apply_result<NoExceptCallable<bool>, volatile copyable_subrange&>();
  test_invalid_apply_result<NoExceptCallable<bool>, const volatile copyable_subrange>();
  test_invalid_apply_result<NoExceptCallable<bool>, const volatile copyable_subrange&>();

  // test tuple
  test_invalid_apply_result_from_function<void(int, char*, double), volatile std::tuple<int, char*, double>>();
  test_invalid_apply_result_from_function<void(int, char*, double), volatile std::tuple<int, char*, double>&>();
  test_invalid_apply_result_from_function<void(int, char*, double), const volatile std::tuple<int, char*, double>>();
  test_invalid_apply_result_from_function<void(int, char*, double), const volatile std::tuple<int, char*, double>&>();
  test_invalid_apply_result_from_function<void(int, char*, double) noexcept, volatile std::tuple<int, char*, double>>();
  test_invalid_apply_result_from_function<void(int, char*, double) noexcept,
                                          volatile std::tuple<int, char*, double>&>();
  test_invalid_apply_result_from_function<void(int, char*, double) noexcept,
                                          const volatile std::tuple<int, char*, double>>();
  test_invalid_apply_result_from_function<void(int, char*, double) noexcept,
                                          const volatile std::tuple<int, char*, double>&>();

  test_invalid_apply_result<NoExceptCallable<bool>, volatile std::tuple<int, char*, double>>();
  test_invalid_apply_result<NoExceptCallable<bool>, volatile std::tuple<int, char*, double>&>();
  test_invalid_apply_result<NoExceptCallable<bool>, const volatile std::tuple<int, char*, double>>();
  test_invalid_apply_result<NoExceptCallable<bool>, const volatile std::tuple<int, char*, double>&>();

  // test pair
  test_invalid_apply_result_from_function<void(int, long), volatile std::pair<int, long>>();
  test_invalid_apply_result_from_function<void(int, long), volatile std::pair<int, long>&>();
  test_invalid_apply_result_from_function<void(int, long), const volatile std::pair<int, long>>();
  test_invalid_apply_result_from_function<void(int, long), const volatile std::pair<int, long>&>();
  test_invalid_apply_result_from_function<void(int, long) noexcept, volatile std::pair<int, long>>();
  test_invalid_apply_result_from_function<void(int, long) noexcept, volatile std::pair<int, long>&>();
  test_invalid_apply_result_from_function<void(int, long) noexcept, const volatile std::pair<int, long>>();
  test_invalid_apply_result_from_function<void(int, long) noexcept, const volatile std::pair<int, long>&>();

  test_invalid_apply_result<NoExceptCallable<bool>, volatile std::pair<int, long>>();
  test_invalid_apply_result<NoExceptCallable<bool>, volatile std::pair<int, long>&>();
  test_invalid_apply_result<NoExceptCallable<bool>, const volatile std::pair<int, long>>();
  test_invalid_apply_result<NoExceptCallable<bool>, const volatile std::pair<int, long>&>();
}

void test_invalid_nontuple_types() {
  // test void
  test_invalid_apply_result_from_function<void(), void>();
  test_invalid_apply_result_from_function<void(), const void>();
  test_invalid_apply_result_from_function<void(), volatile void>();
  test_invalid_apply_result_from_function<void(), const volatile void>();
  test_invalid_apply_result_from_function<void() noexcept, void>();
  test_invalid_apply_result_from_function<void() noexcept, const void>();
  test_invalid_apply_result_from_function<void() noexcept, volatile void>();
  test_invalid_apply_result_from_function<void() noexcept, const volatile void>();

  test_invalid_apply_result<NoExceptCallable<bool>, void>();
  test_invalid_apply_result<NoExceptCallable<bool>, const void>();
  test_invalid_apply_result<NoExceptCallable<bool>, volatile void>();
  test_invalid_apply_result<NoExceptCallable<bool>, const volatile void>();

  // test function
  test_invalid_apply_result_from_function<void(void (&)()), void()>();
  test_invalid_apply_result_from_function<void(void (&)()), void (&)()>();
  test_invalid_apply_result_from_function<void(void (&)()) noexcept, void()>();
  test_invalid_apply_result_from_function<void(void (&)()) noexcept, void (&)()>();
  test_invalid_apply_result_from_function<void(void (&)()), void() const & noexcept>();
  test_invalid_apply_result_from_function<void(void (&)()) noexcept, void() const & noexcept>();

  test_invalid_apply_result<NoExceptCallable<bool>, void()>();
  test_invalid_apply_result<NoExceptCallable<bool>, void (&)()>();
  test_invalid_apply_result<NoExceptCallable<bool>, void() const & noexcept>();

  // test scalar
  test_invalid_apply_result_from_function<void(int), int>();
  test_invalid_apply_result_from_function<void(int), int&>();
  test_invalid_apply_result_from_function<void(int), const int>();
  test_invalid_apply_result_from_function<void(int), const int&>();
  test_invalid_apply_result_from_function<void(int) noexcept, int>();
  test_invalid_apply_result_from_function<void(int) noexcept, int&>();
  test_invalid_apply_result_from_function<void(int) noexcept, const int>();
  test_invalid_apply_result_from_function<void(int) noexcept, const int&>();

  test_invalid_apply_result<NoExceptCallable<bool>, int>();
  test_invalid_apply_result<NoExceptCallable<bool>, int&>();
  test_invalid_apply_result<NoExceptCallable<bool>, const int>();
  test_invalid_apply_result<NoExceptCallable<bool>, const int&>();

  test_invalid_apply_result_from_function<void(void*), void*>();
  test_invalid_apply_result_from_function<void(void*), void*&>();
  test_invalid_apply_result_from_function<void(void*), void* const>();
  test_invalid_apply_result_from_function<void(void*), void* const&>();
  test_invalid_apply_result_from_function<void(void*) noexcept, void*>();
  test_invalid_apply_result_from_function<void(void*) noexcept, void*&>();
  test_invalid_apply_result_from_function<void(void*) noexcept, void* const>();
  test_invalid_apply_result_from_function<void(void*) noexcept, void* const&>();

  test_invalid_apply_result<NoExceptCallable<bool>, void*>();
  test_invalid_apply_result<NoExceptCallable<bool>, void*&>();
  test_invalid_apply_result<NoExceptCallable<bool>, void* const>();
  test_invalid_apply_result<NoExceptCallable<bool>, void* const&>();

  // test plain aggregate
  test_invalid_apply_result_from_function<void(), empty_aggregate>();
  test_invalid_apply_result_from_function<void(), empty_aggregate&>();
  test_invalid_apply_result_from_function<void(), const empty_aggregate>();
  test_invalid_apply_result_from_function<void(), const empty_aggregate&>();
  test_invalid_apply_result_from_function<void() noexcept, empty_aggregate>();
  test_invalid_apply_result_from_function<void() noexcept, empty_aggregate&>();
  test_invalid_apply_result_from_function<void() noexcept, const empty_aggregate>();
  test_invalid_apply_result_from_function<void() noexcept, const empty_aggregate&>();

  test_invalid_apply_result<NoExceptCallable<bool>, empty_aggregate>();
  test_invalid_apply_result<NoExceptCallable<bool>, empty_aggregate&>();
  test_invalid_apply_result<NoExceptCallable<bool>, const empty_aggregate>();
  test_invalid_apply_result<NoExceptCallable<bool>, const empty_aggregate&>();

  // test std::get-able class
  test_invalid_apply_result_from_function<void(int), derived_from_tuple_int>();
  test_invalid_apply_result_from_function<void(int), derived_from_tuple_int&>();
  test_invalid_apply_result_from_function<void(int), const derived_from_tuple_int>();
  test_invalid_apply_result_from_function<void(int), const derived_from_tuple_int&>();
  test_invalid_apply_result_from_function<void(int) noexcept, derived_from_tuple_int>();
  test_invalid_apply_result_from_function<void(int) noexcept, derived_from_tuple_int&>();
  test_invalid_apply_result_from_function<void(int) noexcept, const derived_from_tuple_int>();
  test_invalid_apply_result_from_function<void(int) noexcept, const derived_from_tuple_int&>();

  test_invalid_apply_result<NoExceptCallable<bool>, derived_from_tuple_int>();
  test_invalid_apply_result<NoExceptCallable<bool>, derived_from_tuple_int&>();
  test_invalid_apply_result<NoExceptCallable<bool>, const derived_from_tuple_int>();
  test_invalid_apply_result<NoExceptCallable<bool>, const derived_from_tuple_int&>();

  // test built-in array
  test_invalid_apply_result_from_function<void(int), int[1]>();
  test_invalid_apply_result_from_function<void(int), int (&)[1]>();
  test_invalid_apply_result_from_function<void(int), const int[1]>();
  test_invalid_apply_result_from_function<void(int), const int (&)[1]>();
  test_invalid_apply_result_from_function<void(int) noexcept, int[1]>();
  test_invalid_apply_result_from_function<void(int) noexcept, int (&)[1]>();
  test_invalid_apply_result_from_function<void(int) noexcept, const int[1]>();
  test_invalid_apply_result_from_function<void(int) noexcept, const int (&)[1]>();

  test_invalid_apply_result<NoExceptCallable<bool>, int[1]>();
  test_invalid_apply_result<NoExceptCallable<bool>, int (&)[1]>();
  test_invalid_apply_result<NoExceptCallable<bool>, const int[1]>();
  test_invalid_apply_result<NoExceptCallable<bool>, const int (&)[1]>();
}

void test_invalid_invocations() {
  // test array
  test_invalid_apply_result<void, std::array<int, 0>>();
  test_invalid_apply_result<void*, std::array<int, 1>>();
  test_invalid_apply_result<int, std::array<int, 2>>();
  test_invalid_apply_result<int&, std::array<int, 3>>();
  test_invalid_apply_result<int(int, int, int, int) const & noexcept, std::array<int, 4>>();

  test_invalid_apply_result_from_function<void(int), std::array<int, 0>>();
  test_invalid_apply_result_from_function<void(int), std::array<int, 0>&>();
  test_invalid_apply_result_from_function<void(int), const std::array<int, 0>>();
  test_invalid_apply_result_from_function<void(int), const std::array<int, 0>&>();
  test_invalid_apply_result_from_function<void(int) noexcept, std::array<int, 0>>();
  test_invalid_apply_result_from_function<void(int) noexcept, std::array<int, 0>&>();
  test_invalid_apply_result_from_function<void(int) noexcept, const std::array<int, 0>>();
  test_invalid_apply_result_from_function<void(int) noexcept, const std::array<int, 0>&>();
  test_invalid_apply_result_from_function<void(int, long), std::array<int, 1>>();
  test_invalid_apply_result_from_function<void(int, long), std::array<int, 1>&>();
  test_invalid_apply_result_from_function<void(int, long), const std::array<int, 1>>();
  test_invalid_apply_result_from_function<void(int, long), const std::array<int, 1>&>();
  test_invalid_apply_result_from_function<void(int, long) noexcept, std::array<int, 1>>();
  test_invalid_apply_result_from_function<void(int, long) noexcept, std::array<int, 1>&>();
  test_invalid_apply_result_from_function<void(int, long) noexcept, const std::array<int, 1>>();
  test_invalid_apply_result_from_function<void(int, long) noexcept, const std::array<int, 1>&>();

  // test complex
  test_invalid_apply_result<void, std::complex<float>>();
  test_invalid_apply_result<void*, std::complex<float>>();
  test_invalid_apply_result<int, std::complex<float>>();
  test_invalid_apply_result<int&, std::complex<float>>();
  test_invalid_apply_result<void(float, float) const & noexcept, std::complex<float>>();

  test_invalid_apply_result<void, std::complex<double>>();
  test_invalid_apply_result<void*, std::complex<double>>();
  test_invalid_apply_result<int, std::complex<double>>();
  test_invalid_apply_result<int&, std::complex<double>>();
  test_invalid_apply_result<void(double, double) const & noexcept, std::complex<double>>();

  test_invalid_apply_result<void, std::complex<long double>>();
  test_invalid_apply_result<void*, std::complex<long double>>();
  test_invalid_apply_result<int, std::complex<long double>>();
  test_invalid_apply_result<int&, std::complex<long double>>();
  test_invalid_apply_result<void(long double, long double) const & noexcept, std::complex<long double>>();

  test_invalid_apply_result_from_function<void(float), std::complex<float>>();
  test_invalid_apply_result_from_function<void(float), std::complex<float>&>();
  test_invalid_apply_result_from_function<void(float), const std::complex<float>>();
  test_invalid_apply_result_from_function<void(float), const std::complex<float>&>();
  test_invalid_apply_result_from_function<void(float) noexcept, std::complex<float>>();
  test_invalid_apply_result_from_function<void(float) noexcept, std::complex<float>&>();
  test_invalid_apply_result_from_function<void(float) noexcept, const std::complex<float>>();
  test_invalid_apply_result_from_function<void(float) noexcept, const std::complex<float>&>();

  test_invalid_apply_result_from_function<void(double), std::complex<double>>();
  test_invalid_apply_result_from_function<void(double), std::complex<double>&>();
  test_invalid_apply_result_from_function<void(double), const std::complex<double>>();
  test_invalid_apply_result_from_function<void(double), const std::complex<double>&>();
  test_invalid_apply_result_from_function<void(double) noexcept, std::complex<double>>();
  test_invalid_apply_result_from_function<void(double) noexcept, std::complex<double>&>();
  test_invalid_apply_result_from_function<void(double) noexcept, const std::complex<double>>();
  test_invalid_apply_result_from_function<void(double) noexcept, const std::complex<double>&>();

  test_invalid_apply_result_from_function<void(long double), std::complex<long double>>();
  test_invalid_apply_result_from_function<void(long double), std::complex<long double>&>();
  test_invalid_apply_result_from_function<void(long double), const std::complex<long double>>();
  test_invalid_apply_result_from_function<void(long double), const std::complex<long double>&>();
  test_invalid_apply_result_from_function<void(long double) noexcept, std::complex<long double>>();
  test_invalid_apply_result_from_function<void(long double) noexcept, std::complex<long double>&>();
  test_invalid_apply_result_from_function<void(long double) noexcept, const std::complex<long double>>();
  test_invalid_apply_result_from_function<void(long double) noexcept, const std::complex<long double>&>();

  // test subrange
  using copyable_subrange = std::ranges::subrange<int*>;

  test_invalid_apply_result<void, copyable_subrange>();
  test_invalid_apply_result<void*, copyable_subrange>();
  test_invalid_apply_result<int, copyable_subrange>();
  test_invalid_apply_result<int&, copyable_subrange>();

  test_invalid_apply_result_from_function<void(std::default_sentinel_t), copyable_subrange>();
  test_invalid_apply_result_from_function<void(std::default_sentinel_t), copyable_subrange&>();
  test_invalid_apply_result_from_function<int(std::default_sentinel_t), const copyable_subrange>();
  test_invalid_apply_result_from_function<int(std::default_sentinel_t), const copyable_subrange&>();
  test_invalid_apply_result_from_function<void(std::default_sentinel_t) noexcept, copyable_subrange>();
  test_invalid_apply_result_from_function<void(std::default_sentinel_t) noexcept, copyable_subrange&>();
  test_invalid_apply_result_from_function<int(std::default_sentinel_t) noexcept, const copyable_subrange>();
  test_invalid_apply_result_from_function<int(std::default_sentinel_t) noexcept, const copyable_subrange&>();

  using move_only_counted_iter = std::counted_iterator<cpp20_input_iterator<int*>>;
  using move_only_subrange     = std::ranges::subrange<move_only_counted_iter, std::default_sentinel_t>;

  test_invalid_apply_result<void, move_only_subrange>();
  test_invalid_apply_result<void*, move_only_subrange>();
  test_invalid_apply_result<int, move_only_subrange>();
  test_invalid_apply_result<int&, move_only_subrange>();

  test_invalid_apply_result_from_function<void(move_only_counted_iter&, std::default_sentinel_t),
                                          move_only_subrange&>();
  test_invalid_apply_result_from_function<void(move_only_counted_iter&, std::default_sentinel_t),
                                          const move_only_subrange>();
  test_invalid_apply_result_from_function<void(move_only_counted_iter&, std::default_sentinel_t),
                                          const move_only_subrange&>();
  test_invalid_apply_result_from_function<void(move_only_counted_iter&, std::default_sentinel_t) noexcept,
                                          move_only_subrange&>();
  test_invalid_apply_result_from_function<void(move_only_counted_iter&, std::default_sentinel_t) noexcept,
                                          const move_only_subrange>();
  test_invalid_apply_result_from_function<void(move_only_counted_iter&, std::default_sentinel_t) noexcept,
                                          const move_only_subrange&>();

  test_invalid_apply_result_from_function<void(std::default_sentinel_t), move_only_subrange>();
  test_invalid_apply_result_from_function<void(std::default_sentinel_t), move_only_subrange&>();
  test_invalid_apply_result_from_function<int(std::default_sentinel_t), const move_only_subrange>();
  test_invalid_apply_result_from_function<int(std::default_sentinel_t), const move_only_subrange&>();
  test_invalid_apply_result_from_function<void(std::default_sentinel_t) noexcept, move_only_subrange>();
  test_invalid_apply_result_from_function<void(std::default_sentinel_t) noexcept, move_only_subrange&>();
  test_invalid_apply_result_from_function<int(std::default_sentinel_t) noexcept, const move_only_subrange>();
  test_invalid_apply_result_from_function<int(std::default_sentinel_t) noexcept, const move_only_subrange&>();

  // test tuple
  test_invalid_apply_result<void, std::tuple<>>();
  test_invalid_apply_result<void*, std::tuple<>>();
  test_invalid_apply_result<int, std::tuple<int>>();
  test_invalid_apply_result<int&, std::tuple<int>>();
  test_invalid_apply_result<int() const&, std::tuple<>>();
  test_invalid_apply_result<int() const & noexcept, std::tuple<>>();

  test_invalid_apply_result_from_function<int(), std::tuple<int>>();
  test_invalid_apply_result_from_function<int(), std::tuple<int>&>();
  test_invalid_apply_result_from_function<int(), const std::tuple<int>>();
  test_invalid_apply_result_from_function<int(), const std::tuple<int>&>();
  test_invalid_apply_result_from_function<int() noexcept, std::tuple<int>>();
  test_invalid_apply_result_from_function<int() noexcept, std::tuple<int>&>();
  test_invalid_apply_result_from_function<int() noexcept, const std::tuple<int>>();
  test_invalid_apply_result_from_function<int() noexcept, const std::tuple<int>&>();

  // test pair
  test_invalid_apply_result<void, std::pair<int, long>>();
  test_invalid_apply_result<void*, std::pair<int, long>>();
  test_invalid_apply_result<int, std::pair<int, long>>();
  test_invalid_apply_result<int&, std::pair<int, long>>();

  test_invalid_apply_result<int(int, long) const&, std::pair<int, long>>();
  test_invalid_apply_result<int(int, long) const & noexcept, std::pair<int, long>>();

  test_invalid_apply_result_from_function<int(), std::pair<int, long>>();
  test_invalid_apply_result_from_function<void(int, long, long long), std::pair<int, long>&>();
  test_invalid_apply_result_from_function<int&(int), const std::pair<int, long>>();
  test_invalid_apply_result_from_function<long&(int&, long&), const std::pair<int, long>&>();
  test_invalid_apply_result_from_function<int() noexcept, std::pair<int, long>>();
  test_invalid_apply_result_from_function<void(int, long, long long) noexcept, std::pair<int, long>&>();
  test_invalid_apply_result_from_function<int&(int) noexcept, const std::pair<int, long>>();
  test_invalid_apply_result_from_function<long&(int&, long&) noexcept, const std::pair<int, long>&>();
}
