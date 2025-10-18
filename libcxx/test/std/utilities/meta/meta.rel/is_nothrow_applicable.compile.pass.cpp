//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <type_traits>

// template<class Fn, class Tuple> struct is_nothrow_applicable;

// template<class Fn, class Tuple>
// constexpr bool is_nothrow_applicable_v = is_nothrow_applicable<T, U>::value;

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

template <class Fn, class Tuple, bool Expected>
void test_is_nothrow_applicable() {
  static_assert(std::is_nothrow_applicable<Fn, Tuple>::value == Expected);
  static_assert(std::is_nothrow_applicable_v<Fn, Tuple> == Expected);

  static_assert(std::is_base_of_v<std::bool_constant<Expected>, std::is_nothrow_applicable<Fn, Tuple>>);
  static_assert(std::is_convertible_v<std::is_nothrow_applicable<Fn, Tuple>*, std::bool_constant<Expected>*>);
}

template <class Func, class Tuple, bool Expected>
void test_is_nothrow_applicable_from_function() {
  static_assert(std::is_function_v<Func>);

  test_is_nothrow_applicable<Func, Tuple, Expected>();
  test_is_nothrow_applicable<Func&, Tuple, Expected>();

  test_is_nothrow_applicable<Func*, Tuple, Expected>();
  test_is_nothrow_applicable<Func*&, Tuple, Expected>();
  test_is_nothrow_applicable<Func* const, Tuple, Expected>();
  test_is_nothrow_applicable<Func* volatile, Tuple, Expected>();
  test_is_nothrow_applicable<Func* volatile&, Tuple, Expected>();
  test_is_nothrow_applicable<Func* const volatile, Tuple, Expected>();
  test_is_nothrow_applicable<Func* const volatile&, Tuple, Expected>();
}

void test_valid() {
  // test array
  test_is_nothrow_applicable_from_function<int() noexcept, std::array<int, 0>, true>();
  test_is_nothrow_applicable_from_function<int() noexcept, std::array<long, 0>&, true>();
  test_is_nothrow_applicable_from_function<int() noexcept, const std::array<char, 0>, true>();
  test_is_nothrow_applicable_from_function<int() noexcept, const std::array<std::array<int, 1>, 0>&, true>();

  test_is_nothrow_applicable_from_function<int(long) noexcept, std::array<int, 1>, true>();
  test_is_nothrow_applicable_from_function<int&(int) noexcept, std::array<long, 1>&, true>();
  test_is_nothrow_applicable_from_function<const int&&(float) noexcept, const std::array<double, 1>, true>();
  test_is_nothrow_applicable_from_function<void(double) noexcept, const std::array<char, 1>&, true>();

  test_is_nothrow_applicable_from_function<int(long, int) noexcept, std::array<int, 2>, true>();
  test_is_nothrow_applicable_from_function<int&(long, int) noexcept, std::array<int, 2>&, true>();
  test_is_nothrow_applicable_from_function<const int&&(long, int) noexcept, const std::array<int, 2>, true>();
  test_is_nothrow_applicable_from_function<void(long, int) noexcept, const std::array<int, 2>&, true>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, std::array<int, 0>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, std::array<int, 1>&, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::array<int, 2>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::array<int, 3>&, true>();

  // test complex
  test_is_nothrow_applicable_from_function<float(float, float) noexcept, std::complex<float>, true>();
  test_is_nothrow_applicable_from_function<float(float&, float&) noexcept, std::complex<float>&, true>();
  test_is_nothrow_applicable_from_function<void(float, float) noexcept, const std::complex<float>, true>();
  test_is_nothrow_applicable_from_function<double(float, float) noexcept, const std::complex<float>&, true>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, std::complex<float>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, std::complex<float>&, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::complex<float>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::complex<float>&, true>();

  test_is_nothrow_applicable_from_function<double(double, double) noexcept, std::complex<double>, true>();
  test_is_nothrow_applicable_from_function<double&(double&, double&) noexcept, std::complex<double>&, true>();
  test_is_nothrow_applicable_from_function<void(double, double) noexcept, const std::complex<double>, true>();
  test_is_nothrow_applicable_from_function<double(double, double) noexcept, const std::complex<double>&, true>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, std::complex<double>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, std::complex<double>&, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::complex<double>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::complex<double>&, true>();

  test_is_nothrow_applicable_from_function<long double(long double, long double) noexcept,
                                           std::complex<long double>,
                                           true>();
  test_is_nothrow_applicable_from_function<long double&(long double&, long double&) noexcept,
                                           std::complex<long double>&,
                                           true>();
  test_is_nothrow_applicable_from_function<void(long double, long double) noexcept,
                                           const std::complex<long double>,
                                           true>();
  test_is_nothrow_applicable_from_function<double(long double, long double) noexcept,
                                           const std::complex<long double>&,
                                           true>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, std::complex<long double>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, std::complex<long double>&, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::complex<long double>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::complex<long double>&, true>();

  // test subrange
  // Exception specifications may be different among implementations, see [res.on.exception.handling]/5.
  using copyable_subrange = std::ranges::subrange<int*>;
  constexpr bool can_nothrow_get_copyable_subrange_lv =
      noexcept((void)std::get<0>(std::declval<const copyable_subrange&>()),
               (void)std::get<1>(std::declval<const copyable_subrange&>()));
  constexpr bool can_nothrow_get_copyable_subrange_rv = noexcept(
      (void)std::get<0>(std::declval<copyable_subrange>()), (void)std::get<1>(std::declval<copyable_subrange>()));

  test_is_nothrow_applicable_from_function<void(int*, int*) noexcept,
                                           copyable_subrange,
                                           can_nothrow_get_copyable_subrange_rv>();
  test_is_nothrow_applicable_from_function<long(int*, int*) noexcept,
                                           copyable_subrange&,
                                           can_nothrow_get_copyable_subrange_lv>();
  test_is_nothrow_applicable_from_function<int*&(int*, int*) noexcept,
                                           const copyable_subrange,
                                           can_nothrow_get_copyable_subrange_lv>();
  test_is_nothrow_applicable_from_function<int* && (int*, int*) noexcept,
                                           const copyable_subrange&,
                                           can_nothrow_get_copyable_subrange_lv>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, copyable_subrange, can_nothrow_get_copyable_subrange_rv>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, copyable_subrange&, can_nothrow_get_copyable_subrange_lv>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const copyable_subrange, can_nothrow_get_copyable_subrange_lv>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const copyable_subrange&, can_nothrow_get_copyable_subrange_lv>();

  using move_only_counted_iter = std::counted_iterator<cpp20_input_iterator<int*>>;
  using move_only_subrange     = std::ranges::subrange<move_only_counted_iter, std::default_sentinel_t>;
  constexpr bool can_nothrow_get_move_only_subrange_rv = noexcept(
      (void)std::get<0>(std::declval<move_only_subrange>()), (void)std::get<1>(std::declval<move_only_subrange>()));

  test_is_nothrow_applicable_from_function<void(move_only_counted_iter&&, std::default_sentinel_t),
                                           move_only_subrange,
                                           can_nothrow_get_move_only_subrange_rv>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, move_only_subrange, can_nothrow_get_move_only_subrange_rv>();

  // test tuple
  test_is_nothrow_applicable_from_function<int() noexcept, std::tuple<>, true>();
  test_is_nothrow_applicable_from_function<char&() noexcept, std::tuple<>&, true>();
  test_is_nothrow_applicable_from_function<long&&() noexcept, const std::tuple<>, true>();
  test_is_nothrow_applicable_from_function<void() noexcept, const std::tuple<>&, true>();

  test_is_nothrow_applicable_from_function<int(long, int) noexcept, std::tuple<int, long>, true>();
  test_is_nothrow_applicable_from_function<int&(long, int) noexcept, std::tuple<int, long>&, true>();
  test_is_nothrow_applicable_from_function<const int&&(long, int) noexcept, const std::tuple<int, long>, true>();
  test_is_nothrow_applicable_from_function<void(long, int) noexcept, const std::tuple<int, long>&, true>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, std::tuple<>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, std::tuple<long>&, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::tuple<int, long>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::tuple<int, double, long>&, true>();

  // test pair
  test_is_nothrow_applicable_from_function<int(long, int) noexcept, std::pair<int, long>, true>();
  test_is_nothrow_applicable_from_function<int&(long, int) noexcept, std::pair<int, long>&, true>();
  test_is_nothrow_applicable_from_function<const int&&(long, int) noexcept, const std::pair<int, long>, true>();
  test_is_nothrow_applicable_from_function<void(long, int) noexcept, const std::pair<int, long>&, true>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, std::pair<char, wchar_t>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, std::pair<float, long>&, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::pair<int, long>, true>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const std::pair<int, double>&, true>();
}

void test_potentially_throwing() {
  // test array
  test_is_nothrow_applicable_from_function<int(), std::array<int, 0>, false>();
  test_is_nothrow_applicable_from_function<int(), std::array<long, 0>&, false>();
  test_is_nothrow_applicable_from_function<int(), const std::array<char, 0>, false>();
  test_is_nothrow_applicable_from_function<int(), const std::array<std::array<int, 1>, 0>&, false>();

  test_is_nothrow_applicable_from_function<int(long), std::array<int, 1>, false>();
  test_is_nothrow_applicable_from_function<int&(int), std::array<long, 1>&, false>();
  test_is_nothrow_applicable_from_function<const int&&(float), const std::array<double, 1>, false>();
  test_is_nothrow_applicable_from_function<void(double), const std::array<char, 1>&, false>();

  test_is_nothrow_applicable_from_function<int(long, int), std::array<int, 2>, false>();
  test_is_nothrow_applicable_from_function<int&(long, int), std::array<int, 2>&, false>();
  test_is_nothrow_applicable_from_function<const int&&(long, int), const std::array<int, 2>, false>();
  test_is_nothrow_applicable_from_function<void(long, int), const std::array<int, 2>&, false>();

  test_is_nothrow_applicable<ConstCallable<bool>, std::array<int, 0>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, std::array<int, 1>&, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::array<int, 2>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::array<int, 3>&, false>();

  // test complex
  test_is_nothrow_applicable_from_function<float(float, float), std::complex<float>, false>();
  test_is_nothrow_applicable_from_function<float(float&, float&), std::complex<float>&, false>();
  test_is_nothrow_applicable_from_function<void(float, float), const std::complex<float>, false>();
  test_is_nothrow_applicable_from_function<double(float, float), const std::complex<float>&, false>();

  test_is_nothrow_applicable<ConstCallable<bool>, std::complex<float>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, std::complex<float>&, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::complex<float>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::complex<float>&, false>();

  test_is_nothrow_applicable_from_function<double(double, double), std::complex<double>, false>();
  test_is_nothrow_applicable_from_function<double&(double&, double&), std::complex<double>&, false>();
  test_is_nothrow_applicable_from_function<void(double, double), const std::complex<double>, false>();
  test_is_nothrow_applicable_from_function<double(double, double), std::complex<double>&, false>();

  test_is_nothrow_applicable<ConstCallable<bool>, std::complex<double>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, std::complex<double>&, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::complex<double>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::complex<double>&, false>();

  test_is_nothrow_applicable_from_function<long double(long double, long double), std::complex<long double>, false>();
  test_is_nothrow_applicable_from_function<long double&(long double&, long double&),
                                           std::complex<long double>&,
                                           false>();
  test_is_nothrow_applicable_from_function<void(long double, long double), const std::complex<long double>, false>();
  test_is_nothrow_applicable_from_function<double(long double, long double), const std::complex<long double>&, false>();

  test_is_nothrow_applicable<ConstCallable<bool>, std::complex<long double>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, std::complex<long double>&, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::complex<long double>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::complex<long double>&, false>();

  // test subrange
  using copyable_subrange = std::ranges::subrange<int*>;
  test_is_nothrow_applicable_from_function<void(int*, int*), copyable_subrange, false>();
  test_is_nothrow_applicable_from_function<long(int*, int*), copyable_subrange&, false>();
  test_is_nothrow_applicable_from_function<int*&(int*, int*), const copyable_subrange, false>();
  test_is_nothrow_applicable_from_function<int* && (int*, int*), const copyable_subrange&, false>();

  test_is_nothrow_applicable<ConstCallable<bool>, copyable_subrange, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, copyable_subrange&, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const copyable_subrange, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const copyable_subrange&, false>();

  using move_only_counted_iter = std::counted_iterator<cpp20_input_iterator<int*>>;
  using move_only_subrange     = std::ranges::subrange<move_only_counted_iter, std::default_sentinel_t>;
  test_is_nothrow_applicable_from_function<void(move_only_counted_iter&&, std::default_sentinel_t),
                                           move_only_subrange,
                                           false>();

  test_is_nothrow_applicable<ConstCallable<bool>, move_only_subrange, false>();

  // test tuple
  test_is_nothrow_applicable_from_function<int(), std::tuple<>, false>();
  test_is_nothrow_applicable_from_function<char&(), std::tuple<>&, false>();
  test_is_nothrow_applicable_from_function<long&&(), const std::tuple<>, false>();
  test_is_nothrow_applicable_from_function<void(), const std::tuple<>&, false>();

  test_is_nothrow_applicable_from_function<int(long, int), std::tuple<int, long>, false>();
  test_is_nothrow_applicable_from_function<int&(long, int), std::tuple<int, long>&, false>();
  test_is_nothrow_applicable_from_function<const int&&(long, int), const std::tuple<int, long>, false>();
  test_is_nothrow_applicable_from_function<void(long, int), const std::tuple<int, long>&, false>();

  test_is_nothrow_applicable<ConstCallable<bool>, std::tuple<>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, std::tuple<long>&, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::tuple<int, long>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::tuple<int, double, long>&, false>();

  // test pair
  test_is_nothrow_applicable_from_function<int(long, int), std::pair<int, long>, false>();
  test_is_nothrow_applicable_from_function<int&(long, int), std::pair<int, long>&, false>();
  test_is_nothrow_applicable_from_function<const int&&(long, int), const std::pair<int, long>, false>();
  test_is_nothrow_applicable_from_function<void(long, int), const std::pair<int, long>&, false>();

  test_is_nothrow_applicable<ConstCallable<bool>, std::pair<char, wchar_t>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, std::pair<float, long>&, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::pair<int, long>, false>();
  test_is_nothrow_applicable<ConstCallable<bool>, const std::pair<int, double>&, false>();
}

void test_volatile() {
  // test array
  test_is_nothrow_applicable_from_function<void(int), volatile std::array<int, 1>, false>();
  test_is_nothrow_applicable_from_function<void(int), volatile std::array<int, 1>&, false>();
  test_is_nothrow_applicable_from_function<void(int), const volatile std::array<int, 1>, false>();
  test_is_nothrow_applicable_from_function<void(int), const volatile std::array<int, 1>&, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, volatile std::array<int, 1>, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, volatile std::array<int, 1>&, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, const volatile std::array<int, 1>, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, const volatile std::array<int, 1>&, false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, volatile std::array<int, 1>, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, volatile std::array<int, 1>&, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const volatile std::array<int, 1>, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const volatile std::array<int, 1>&, false>();

  // test complex
  test_is_nothrow_applicable_from_function<void(double, double), volatile std::complex<double>, false>();
  test_is_nothrow_applicable_from_function<void(double, double), volatile std::complex<double>&, false>();
  test_is_nothrow_applicable_from_function<void(double, double), const volatile std::complex<double>, false>();
  test_is_nothrow_applicable_from_function<void(double, double), const volatile std::complex<double>&, false>();
  test_is_nothrow_applicable_from_function<void(double, double) noexcept, volatile std::complex<double>, false>();
  test_is_nothrow_applicable_from_function<void(double, double) noexcept, volatile std::complex<double>&, false>();
  test_is_nothrow_applicable_from_function<void(double, double) noexcept, const volatile std::complex<double>, false>();
  test_is_nothrow_applicable_from_function<void(double, double) noexcept,
                                           const volatile std::complex<double>&,
                                           false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, volatile std::complex<double>, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, volatile std::complex<double>&, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const volatile std::complex<double>, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const volatile std::complex<double>&, false>();

  // test subrange
  using copyable_subrange = std::ranges::subrange<int*>;

  test_is_nothrow_applicable_from_function<void(int*, int*), volatile copyable_subrange, false>();
  test_is_nothrow_applicable_from_function<void(int*, int*), volatile copyable_subrange&, false>();
  test_is_nothrow_applicable_from_function<void(int*, int*), const volatile copyable_subrange, false>();
  test_is_nothrow_applicable_from_function<void(int*, int*), const volatile copyable_subrange&, false>();
  test_is_nothrow_applicable_from_function<void(int*, int*) noexcept, volatile copyable_subrange, false>();
  test_is_nothrow_applicable_from_function<void(int*, int*) noexcept, volatile copyable_subrange&, false>();
  test_is_nothrow_applicable_from_function<void(int*, int*) noexcept, const volatile copyable_subrange, false>();
  test_is_nothrow_applicable_from_function<void(int*, int*) noexcept, const volatile copyable_subrange&, false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, volatile copyable_subrange, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, volatile copyable_subrange&, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const volatile copyable_subrange, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const volatile copyable_subrange&, false>();

  // test tuple
  test_is_nothrow_applicable_from_function<void(int, char*, double), volatile std::tuple<int, char*, double>, false>();
  test_is_nothrow_applicable_from_function<void(int, char*, double), volatile std::tuple<int, char*, double>&, false>();
  test_is_nothrow_applicable_from_function<void(int, char*, double),
                                           const volatile std::tuple<int, char*, double>,
                                           false>();
  test_is_nothrow_applicable_from_function<void(int, char*, double),
                                           const volatile std::tuple<int, char*, double>&,
                                           false>();
  test_is_nothrow_applicable_from_function<void(int, char*, double) noexcept,
                                           volatile std::tuple<int, char*, double>,
                                           false>();
  test_is_nothrow_applicable_from_function<void(int, char*, double) noexcept,
                                           volatile std::tuple<int, char*, double>&,
                                           false>();
  test_is_nothrow_applicable_from_function<void(int, char*, double) noexcept,
                                           const volatile std::tuple<int, char*, double>,
                                           false>();
  test_is_nothrow_applicable_from_function<void(int, char*, double) noexcept,
                                           const volatile std::tuple<int, char*, double>&,
                                           false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, volatile std::tuple<int, char*, double>, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, volatile std::tuple<int, char*, double>&, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const volatile std::tuple<int, char*, double>, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const volatile std::tuple<int, char*, double>&, false>();

  // test pair
  test_is_nothrow_applicable_from_function<void(int, long), volatile std::pair<int, long>, false>();
  test_is_nothrow_applicable_from_function<void(int, long), volatile std::pair<int, long>&, false>();
  test_is_nothrow_applicable_from_function<void(int, long), const volatile std::pair<int, long>, false>();
  test_is_nothrow_applicable_from_function<void(int, long), const volatile std::pair<int, long>&, false>();
  test_is_nothrow_applicable_from_function<void(int, long) noexcept, volatile std::pair<int, long>, false>();
  test_is_nothrow_applicable_from_function<void(int, long) noexcept, volatile std::pair<int, long>&, false>();
  test_is_nothrow_applicable_from_function<void(int, long) noexcept, const volatile std::pair<int, long>, false>();
  test_is_nothrow_applicable_from_function<void(int, long) noexcept, const volatile std::pair<int, long>&, false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, volatile std::pair<int, long>, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, volatile std::pair<int, long>&, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const volatile std::pair<int, long>, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const volatile std::pair<int, long>&, false>();
}

void test_invalid_nontuple_types() {
  // test void
  test_is_nothrow_applicable_from_function<void(), void, false>();
  test_is_nothrow_applicable_from_function<void(), const void, false>();
  test_is_nothrow_applicable_from_function<void(), volatile void, false>();
  test_is_nothrow_applicable_from_function<void(), const volatile void, false>();
  test_is_nothrow_applicable_from_function<void() noexcept, void, false>();
  test_is_nothrow_applicable_from_function<void() noexcept, const void, false>();
  test_is_nothrow_applicable_from_function<void() noexcept, volatile void, false>();
  test_is_nothrow_applicable_from_function<void() noexcept, const volatile void, false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, void, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const void, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, volatile void, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const volatile void, false>();

  // test function
  test_is_nothrow_applicable_from_function<void(void (&)()), void(), false>();
  test_is_nothrow_applicable_from_function<void(void (&)()), void (&)(), false>();
  test_is_nothrow_applicable_from_function<void(void (&)()) noexcept, void(), false>();
  test_is_nothrow_applicable_from_function<void(void (&)()) noexcept, void (&)(), false>();
  test_is_nothrow_applicable_from_function<void(void (&)()), void() const & noexcept, false>();
  test_is_nothrow_applicable_from_function<void(void (&)()) noexcept, void() const & noexcept, false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, void(), false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, void (&)(), false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, void() const & noexcept, false>();

  // test scalar
  test_is_nothrow_applicable_from_function<void(int), int, false>();
  test_is_nothrow_applicable_from_function<void(int), int&, false>();
  test_is_nothrow_applicable_from_function<void(int), const int, false>();
  test_is_nothrow_applicable_from_function<void(int), const int&, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, int, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, int&, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, const int, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, const int&, false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, int, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, int&, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const int, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const int&, false>();

  test_is_nothrow_applicable_from_function<void(void*), void*, false>();
  test_is_nothrow_applicable_from_function<void(void*), void*&, false>();
  test_is_nothrow_applicable_from_function<void(void*), void* const, false>();
  test_is_nothrow_applicable_from_function<void(void*), void* const&, false>();
  test_is_nothrow_applicable_from_function<void(void*) noexcept, void*, false>();
  test_is_nothrow_applicable_from_function<void(void*) noexcept, void*&, false>();
  test_is_nothrow_applicable_from_function<void(void*) noexcept, void* const, false>();
  test_is_nothrow_applicable_from_function<void(void*) noexcept, void* const&, false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, void*, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, void*&, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, void* const, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, void* const&, false>();

  // test plain aggregate
  test_is_nothrow_applicable_from_function<void(), empty_aggregate, false>();
  test_is_nothrow_applicable_from_function<void(), empty_aggregate&, false>();
  test_is_nothrow_applicable_from_function<void(), const empty_aggregate, false>();
  test_is_nothrow_applicable_from_function<void(), const empty_aggregate&, false>();
  test_is_nothrow_applicable_from_function<void() noexcept, empty_aggregate, false>();
  test_is_nothrow_applicable_from_function<void() noexcept, empty_aggregate&, false>();
  test_is_nothrow_applicable_from_function<void() noexcept, const empty_aggregate, false>();
  test_is_nothrow_applicable_from_function<void() noexcept, const empty_aggregate&, false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, empty_aggregate, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, empty_aggregate&, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const empty_aggregate, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const empty_aggregate&, false>();

  // test std::get-able class
  test_is_nothrow_applicable_from_function<void(int), derived_from_tuple_int, false>();
  test_is_nothrow_applicable_from_function<void(int), derived_from_tuple_int&, false>();
  test_is_nothrow_applicable_from_function<void(int), const derived_from_tuple_int, false>();
  test_is_nothrow_applicable_from_function<void(int), const derived_from_tuple_int&, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, derived_from_tuple_int, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, derived_from_tuple_int&, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, const derived_from_tuple_int, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, const derived_from_tuple_int&, false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, derived_from_tuple_int, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, derived_from_tuple_int&, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const derived_from_tuple_int, false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const derived_from_tuple_int&, false>();

  // test built-in array
  test_is_nothrow_applicable_from_function<void(int), int[1], false>();
  test_is_nothrow_applicable_from_function<void(int), int (&)[1], false>();
  test_is_nothrow_applicable_from_function<void(int), const int[1], false>();
  test_is_nothrow_applicable_from_function<void(int), const int (&)[1], false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, int[1], false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, int (&)[1], false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, const int[1], false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, const int (&)[1], false>();

  test_is_nothrow_applicable<NoExceptCallable<bool>, int[1], false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, int (&)[1], false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const int[1], false>();
  test_is_nothrow_applicable<NoExceptCallable<bool>, const int (&)[1], false>();
}

void test_invalid_invocations() {
  // test array
  test_is_nothrow_applicable<void, std::array<int, 0>, false>();
  test_is_nothrow_applicable<void*, std::array<int, 1>, false>();
  test_is_nothrow_applicable<int, std::array<int, 2>, false>();
  test_is_nothrow_applicable<int&, std::array<int, 3>, false>();
  test_is_nothrow_applicable<int(int, int, int, int) const & noexcept, std::array<int, 4>, false>();

  test_is_nothrow_applicable_from_function<void(int), std::array<int, 0>, false>();
  test_is_nothrow_applicable_from_function<void(int), std::array<int, 0>&, false>();
  test_is_nothrow_applicable_from_function<void(int), const std::array<int, 0>, false>();
  test_is_nothrow_applicable_from_function<void(int), const std::array<int, 0>&, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, std::array<int, 0>, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, std::array<int, 0>&, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, const std::array<int, 0>, false>();
  test_is_nothrow_applicable_from_function<void(int) noexcept, const std::array<int, 0>&, false>();
  test_is_nothrow_applicable_from_function<void(int, long), std::array<int, 1>, false>();
  test_is_nothrow_applicable_from_function<void(int, long), std::array<int, 1>&, false>();
  test_is_nothrow_applicable_from_function<void(int, long), const std::array<int, 1>, false>();
  test_is_nothrow_applicable_from_function<void(int, long), const std::array<int, 1>&, false>();
  test_is_nothrow_applicable_from_function<void(int, long) noexcept, std::array<int, 1>, false>();
  test_is_nothrow_applicable_from_function<void(int, long) noexcept, std::array<int, 1>&, false>();
  test_is_nothrow_applicable_from_function<void(int, long) noexcept, const std::array<int, 1>, false>();
  test_is_nothrow_applicable_from_function<void(int, long) noexcept, const std::array<int, 1>&, false>();

  // test complex
  test_is_nothrow_applicable<void, std::complex<float>, false>();
  test_is_nothrow_applicable<void*, std::complex<float>, false>();
  test_is_nothrow_applicable<int, std::complex<float>, false>();
  test_is_nothrow_applicable<int&, std::complex<float>, false>();
  test_is_nothrow_applicable<void(float, float) const & noexcept, std::complex<float>, false>();

  test_is_nothrow_applicable<void, std::complex<double>, false>();
  test_is_nothrow_applicable<void*, std::complex<double>, false>();
  test_is_nothrow_applicable<int, std::complex<double>, false>();
  test_is_nothrow_applicable<int&, std::complex<double>, false>();
  test_is_nothrow_applicable<void(double, double) const & noexcept, std::complex<double>, false>();

  test_is_nothrow_applicable<void, std::complex<long double>, false>();
  test_is_nothrow_applicable<void*, std::complex<long double>, false>();
  test_is_nothrow_applicable<int, std::complex<long double>, false>();
  test_is_nothrow_applicable<int&, std::complex<long double>, false>();
  test_is_nothrow_applicable<void(long double, long double) const & noexcept, std::complex<long double>, false>();

  test_is_nothrow_applicable_from_function<void(float), std::complex<float>, false>();
  test_is_nothrow_applicable_from_function<void(float), std::complex<float>&, false>();
  test_is_nothrow_applicable_from_function<void(float), const std::complex<float>, false>();
  test_is_nothrow_applicable_from_function<void(float), const std::complex<float>&, false>();
  test_is_nothrow_applicable_from_function<void(float) noexcept, std::complex<float>, false>();
  test_is_nothrow_applicable_from_function<void(float) noexcept, std::complex<float>&, false>();
  test_is_nothrow_applicable_from_function<void(float) noexcept, const std::complex<float>, false>();
  test_is_nothrow_applicable_from_function<void(float) noexcept, const std::complex<float>&, false>();

  test_is_nothrow_applicable_from_function<void(double), std::complex<double>, false>();
  test_is_nothrow_applicable_from_function<void(double), std::complex<double>&, false>();
  test_is_nothrow_applicable_from_function<void(double), const std::complex<double>, false>();
  test_is_nothrow_applicable_from_function<void(double), const std::complex<double>&, false>();
  test_is_nothrow_applicable_from_function<void(double) noexcept, std::complex<double>, false>();
  test_is_nothrow_applicable_from_function<void(double) noexcept, std::complex<double>&, false>();
  test_is_nothrow_applicable_from_function<void(double) noexcept, const std::complex<double>, false>();
  test_is_nothrow_applicable_from_function<void(double) noexcept, const std::complex<double>&, false>();

  test_is_nothrow_applicable_from_function<void(long double), std::complex<long double>, false>();
  test_is_nothrow_applicable_from_function<void(long double), std::complex<long double>&, false>();
  test_is_nothrow_applicable_from_function<void(long double), const std::complex<long double>, false>();
  test_is_nothrow_applicable_from_function<void(long double), const std::complex<long double>&, false>();
  test_is_nothrow_applicable_from_function<void(long double) noexcept, std::complex<long double>, false>();
  test_is_nothrow_applicable_from_function<void(long double) noexcept, std::complex<long double>&, false>();
  test_is_nothrow_applicable_from_function<void(long double) noexcept, const std::complex<long double>, false>();
  test_is_nothrow_applicable_from_function<void(long double) noexcept, const std::complex<long double>&, false>();

  // test subrange
  using copyable_subrange = std::ranges::subrange<int*>;

  test_is_nothrow_applicable<void, copyable_subrange, false>();
  test_is_nothrow_applicable<void*, copyable_subrange, false>();
  test_is_nothrow_applicable<int, copyable_subrange, false>();
  test_is_nothrow_applicable<int&, copyable_subrange, false>();

  test_is_nothrow_applicable_from_function<void(std::default_sentinel_t), copyable_subrange, false>();
  test_is_nothrow_applicable_from_function<void(std::default_sentinel_t), copyable_subrange&, false>();
  test_is_nothrow_applicable_from_function<int(std::default_sentinel_t), const copyable_subrange, false>();
  test_is_nothrow_applicable_from_function<int(std::default_sentinel_t), const copyable_subrange&, false>();
  test_is_nothrow_applicable_from_function<void(std::default_sentinel_t) noexcept, copyable_subrange, false>();
  test_is_nothrow_applicable_from_function<void(std::default_sentinel_t) noexcept, copyable_subrange&, false>();
  test_is_nothrow_applicable_from_function<int(std::default_sentinel_t) noexcept, const copyable_subrange, false>();
  test_is_nothrow_applicable_from_function<int(std::default_sentinel_t) noexcept, const copyable_subrange&, false>();

  using move_only_counted_iter = std::counted_iterator<cpp20_input_iterator<int*>>;
  using move_only_subrange     = std::ranges::subrange<move_only_counted_iter, std::default_sentinel_t>;

  test_is_nothrow_applicable<void, move_only_subrange, false>();
  test_is_nothrow_applicable<void*, move_only_subrange, false>();
  test_is_nothrow_applicable<int, move_only_subrange, false>();
  test_is_nothrow_applicable<int&, move_only_subrange, false>();

  test_is_nothrow_applicable_from_function<void(move_only_counted_iter&, std::default_sentinel_t),
                                           move_only_subrange&,
                                           false>();
  test_is_nothrow_applicable_from_function<void(move_only_counted_iter&, std::default_sentinel_t),
                                           const move_only_subrange,
                                           false>();
  test_is_nothrow_applicable_from_function<void(move_only_counted_iter&, std::default_sentinel_t),
                                           const move_only_subrange&,
                                           false>();
  test_is_nothrow_applicable_from_function<void(move_only_counted_iter&, std::default_sentinel_t) noexcept,
                                           move_only_subrange&,
                                           false>();
  test_is_nothrow_applicable_from_function<void(move_only_counted_iter&, std::default_sentinel_t) noexcept,
                                           const move_only_subrange,
                                           false>();
  test_is_nothrow_applicable_from_function<void(move_only_counted_iter&, std::default_sentinel_t) noexcept,
                                           const move_only_subrange&,
                                           false>();

  test_is_nothrow_applicable_from_function<void(std::default_sentinel_t), move_only_subrange, false>();
  test_is_nothrow_applicable_from_function<void(std::default_sentinel_t), move_only_subrange&, false>();
  test_is_nothrow_applicable_from_function<int(std::default_sentinel_t), const move_only_subrange, false>();
  test_is_nothrow_applicable_from_function<int(std::default_sentinel_t), const move_only_subrange&, false>();
  test_is_nothrow_applicable_from_function<void(std::default_sentinel_t) noexcept, move_only_subrange, false>();
  test_is_nothrow_applicable_from_function<void(std::default_sentinel_t) noexcept, move_only_subrange&, false>();
  test_is_nothrow_applicable_from_function<int(std::default_sentinel_t) noexcept, const move_only_subrange, false>();
  test_is_nothrow_applicable_from_function<int(std::default_sentinel_t) noexcept, const move_only_subrange&, false>();

  // test tuple
  test_is_nothrow_applicable<void, std::tuple<>, false>();
  test_is_nothrow_applicable<void*, std::tuple<>, false>();
  test_is_nothrow_applicable<int, std::tuple<int>, false>();
  test_is_nothrow_applicable<int&, std::tuple<int>, false>();
  test_is_nothrow_applicable<int() const&, std::tuple<>, false>();
  test_is_nothrow_applicable<int() const & noexcept, std::tuple<>, false>();

  test_is_nothrow_applicable_from_function<int(), std::tuple<int>, false>();
  test_is_nothrow_applicable_from_function<int(), std::tuple<int>&, false>();
  test_is_nothrow_applicable_from_function<int(), const std::tuple<int>, false>();
  test_is_nothrow_applicable_from_function<int(), const std::tuple<int>&, false>();
  test_is_nothrow_applicable_from_function<int() noexcept, std::tuple<int>, false>();
  test_is_nothrow_applicable_from_function<int() noexcept, std::tuple<int>&, false>();
  test_is_nothrow_applicable_from_function<int() noexcept, const std::tuple<int>, false>();
  test_is_nothrow_applicable_from_function<int() noexcept, const std::tuple<int>&, false>();

  // test pair
  test_is_nothrow_applicable<void, std::pair<int, long>, false>();
  test_is_nothrow_applicable<void*, std::pair<int, long>, false>();
  test_is_nothrow_applicable<int, std::pair<int, long>, false>();
  test_is_nothrow_applicable<int&, std::pair<int, long>, false>();

  test_is_nothrow_applicable<int(int, long) const&, std::pair<int, long>, false>();
  test_is_nothrow_applicable<int(int, long) const & noexcept, std::pair<int, long>, false>();

  test_is_nothrow_applicable_from_function<int(), std::pair<int, long>, false>();
  test_is_nothrow_applicable_from_function<void(int, long, long long), std::pair<int, long>&, false>();
  test_is_nothrow_applicable_from_function<int&(int), const std::pair<int, long>, false>();
  test_is_nothrow_applicable_from_function<long&(int&, long&), const std::pair<int, long>&, false>();
  test_is_nothrow_applicable_from_function<int() noexcept, std::pair<int, long>, false>();
  test_is_nothrow_applicable_from_function<void(int, long, long long) noexcept, std::pair<int, long>&, false>();
  test_is_nothrow_applicable_from_function<int&(int) noexcept, const std::pair<int, long>, false>();
  test_is_nothrow_applicable_from_function<long&(int&, long&) noexcept, const std::pair<int, long>&, false>();
}
