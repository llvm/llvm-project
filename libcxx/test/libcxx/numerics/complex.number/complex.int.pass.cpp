//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// [complex.numbers.general]/2:
// The effect of instantiating the primary template of complex for any type that is not a cv-unqualified floating-point
// type ([basic.fundamental]) is unspecified. Specializations of complex for cv-unqualified floating-point types are
// trivially copyable literal types ([basic.types.general]).

// libc++ supports the following operations and properties of complex<Integral> as extensions.
// - Construction, assignment, and trivial copyability.
// - std::real()/std::imag() and member real()/imag() functions.
// - Addition, subtraction, multiplication, and division.

// (unsigned) _BitInt(N) are not covered when N > 64 due to the following bugs.
// TODO: Expand the test coverage once all related bugs a fixed.
// - https://llvm.org/PR196107: Widths larger than 64 are problematic on some platforms due to lack of intrinsics.
// - https://llvm.org/PR242211: Widths larger than 128 can cause Clang assertion builds to crash.

#include <cassert>
#include <complex>
#include <type_traits>

#include "test_macros.h"

template <class T>
TEST_CONSTEXPR_CXX14 void test_0() {
  std::complex<T> z;

  ASSERT_SAME_TYPE(decltype(z.real()), T);
  ASSERT_SAME_TYPE(decltype(std::real(z)), T);
  assert(z.real() == 0);
  assert(std::real(z) == 0);

  ASSERT_SAME_TYPE(decltype(z.imag()), T);
  ASSERT_SAME_TYPE(decltype(std::imag(z)), T);
  assert(z.imag() == 0);
  assert(std::imag(z) == 0);
}

template <class T>
TEST_CONSTEXPR_CXX14 void test_1(T x) {
  std::complex<T> z(x);

  assert(z.real() == x);
  assert(std::real(z) == x);

  assert(z.imag() == 0);
  assert(std::imag(z) == 0);
}

template <class T>
TEST_CONSTEXPR_CXX14 void test_2(T x, T y) {
  std::complex<T> z(x, y);

  assert(z.real() == x);
  assert(std::real(z) == x);

  assert(z.imag() == y);
  assert(std::imag(z) == y);
}

template <class F, class I>
TEST_CONSTEXPR_CXX20 void test_compound_assignment() {
  {
    std::complex<I> p(4, 2);
    std::complex<I> q(1, 7);

    ASSERT_SAME_TYPE(decltype(p += q), std::complex<I>&);
    std::complex<I>& result = p += q;
    assert(result == std::complex<I>(5, 9));
    assert(&result == &p);
  }
  {
    std::complex<F> p(4.0f, 2.0f);
    std::complex<I> q(1, 7);

    ASSERT_SAME_TYPE(decltype(p += q), std::complex<F>&);
    std::complex<F>& result = p += q;
    assert(result == std::complex<F>(5.0f, 9.0f));
    assert(&result == &p);
  }
  {
    std::complex<I> p(4, 2);
    std::complex<F> q(1.0f, 7.0f);

    ASSERT_SAME_TYPE(decltype(p += q), std::complex<I>&);
    std::complex<I>& result = p += q;
    assert(result == std::complex<I>(5, 9));
    assert(&result == &p);
  }

  {
    std::complex<I> p(17, 29);
    std::complex<I> q(4, 2);

    ASSERT_SAME_TYPE(decltype(p -= q), std::complex<I>&);
    std::complex<I>& result = p -= q;
    assert(result == std::complex<I>(13, 27));
    assert(&result == &p);
  }
  {
    std::complex<F> p(17.0f, 29.0f);
    std::complex<I> q(4, 2);

    ASSERT_SAME_TYPE(decltype(p -= q), std::complex<F>&);
    std::complex<F>& result = p -= q;
    assert(result == std::complex<F>(13.0f, 27.0f));
    assert(&result == &p);
  }
  {
    std::complex<I> p(17, 29);
    std::complex<F> q(4.0f, 2.0f);

    ASSERT_SAME_TYPE(decltype(p -= q), std::complex<I>&);
    std::complex<I>& result = p -= q;
    assert(result == std::complex<I>(13, 27));
    assert(&result == &p);
  }

  {
    std::complex<I> p(3, 2);
    std::complex<I> q(4, 1);

    ASSERT_SAME_TYPE(decltype(p *= q), std::complex<I>&);
    std::complex<I>& result = p *= q;
    assert(result == std::complex<I>(10, 11));
    assert(&result == &p);
  }
  {
    std::complex<F> p(3.0f, 2.0f);
    std::complex<I> q(4, 1);

    ASSERT_SAME_TYPE(decltype(p *= q), std::complex<F>&);
    std::complex<F>& result = p *= q;
    assert(result == std::complex<F>(10.0f, 11.0f));
    assert(&result == &p);
  }
  {
    std::complex<I> p(3, 2);
    std::complex<F> q(4.0f, 1.0f);

    ASSERT_SAME_TYPE(decltype(p *= q), std::complex<I>&);
    std::complex<I>& result = p *= q;
    assert(result == std::complex<I>(10, 11));
    assert(&result == &p);
  }

  {
    std::complex<I> p(2, 2);
    std::complex<I> q(1, 1);

    ASSERT_SAME_TYPE(decltype(p /= q), std::complex<I>&);
    std::complex<I>& result = p /= q;
    assert(result == std::complex<I>(2, 0));
    assert(&result == &p);
  }
  {
    std::complex<F> p(2.0f, 2.0f);
    std::complex<I> q(1, 1);

    ASSERT_SAME_TYPE(decltype(p /= q), std::complex<F>&);
    std::complex<F>& result = p /= q;
    assert(result == std::complex<F>(2.0f, 0));
    assert(&result == &p);
  }
  {
    std::complex<I> p(2, 2);
    std::complex<F> q(1.0f, 1.0f);

    ASSERT_SAME_TYPE(decltype(p /= q), std::complex<I>&);
    std::complex<I>& result = p /= q;
    assert(result == std::complex<I>(2, 0));
    assert(&result == &p);
  }
}

template <class T>
TEST_CONSTEXPR_CXX14 void test_type_properties() {
  static_assert(std::is_trivially_copyable<std::complex<T> >::value, "");
  static_assert(std::is_trivially_copy_constructible<std::complex<T> >::value, "");
  static_assert(std::is_trivially_move_constructible<std::complex<T> >::value, "");
  static_assert(std::is_trivially_copy_assignable<std::complex<T> >::value, "");
  static_assert(std::is_trivially_move_assignable<std::complex<T> >::value, "");
  static_assert(std::is_trivially_destructible<std::complex<T> >::value, "");
}

template <class T>
TEST_CONSTEXPR_CXX14 void test() {
  test_type_properties<T>();

  test_0<T>();

  test_1<T>(0);
  test_1<T>(1);
  test_1<T>(10);

  test_2<T>(0, 0);
  test_2<T>(3, 0);
  test_2<T>(0, 4);
  test_2<T>(3, 4);

  if (TEST_STD_AT_LEAST_20_OR_RUNTIME_EVALUATED) {
    test_compound_assignment<float, T>();
    test_compound_assignment<double, T>();
    test_compound_assignment<long double, T>();
  }
}

TEST_CONSTEXPR_CXX14 bool test() {
  test<signed char>();
  test<short>();
  test<int>();
  test<long>();
  test<long long>();
#ifndef TEST_HAS_NO_INT128
  test<__int128_t>();
#endif
#if TEST_HAS_BITINT
  test<_BitInt(7)>();
  test<_BitInt(8)>();
  test<_BitInt(9)>();
  test<_BitInt(15)>();
  test<_BitInt(16)>();
  test<_BitInt(17)>();
  test<_BitInt(31)>();
  test<_BitInt(32)>();
  test<_BitInt(33)>();
  test<_BitInt(48)>();
  test<_BitInt(63)>();
  test<_BitInt(64)>();
#endif

  test<unsigned char>();
  test<unsigned short>();
  test<unsigned int>();
  test<unsigned long>();
  test<unsigned long long>();
#ifndef TEST_HAS_NO_INT128
  test<__uint128_t>();
#endif
#if TEST_HAS_BITINT
  test<unsigned _BitInt(7)>();
  test<unsigned _BitInt(8)>();
  test<unsigned _BitInt(9)>();
  test<unsigned _BitInt(15)>();
  test<unsigned _BitInt(16)>();
  test<unsigned _BitInt(17)>();
  test<unsigned _BitInt(31)>();
  test<unsigned _BitInt(32)>();
  test<unsigned _BitInt(33)>();
  test<unsigned _BitInt(48)>();
  test<unsigned _BitInt(63)>();
  test<unsigned _BitInt(64)>();
#endif

  test<char>();
#ifndef TEST_HAS_NO_CHAR8_T
  test<char8_t>();
#endif
  test<char16_t>();
  test<char32_t>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  // Test types that are too narrow to hold consistent arithmetic results.
  {
    test_type_properties<bool>();

    test_0<bool>();

    test_1<bool>(false);
    test_1<bool>(true);

    test_2<bool>(false, false);
    test_2<bool>(false, true);
    test_2<bool>(true, false);
    test_2<bool>(true, true);
  }
#if TEST_HAS_BITINT
  {
    test_type_properties<_BitInt(2)>();

    test_0<_BitInt(2)>();

    test_1<_BitInt(2)>(0);
    test_1<_BitInt(2)>(-2);

    test_2<_BitInt(2)>(0, 0);
    test_2<_BitInt(2)>(-2, 0);
    test_2<_BitInt(2)>(0, -1);
    test_2<_BitInt(2)>(-2, -1);
  }
  {
    test_type_properties<unsigned _BitInt(1)>();

    test_0<unsigned _BitInt(1)>();

    test_1<unsigned _BitInt(1)>(0);
    test_1<unsigned _BitInt(1)>(1);

    test_2<unsigned _BitInt(1)>(0, 0);
    test_2<unsigned _BitInt(1)>(1, 0);
    test_2<unsigned _BitInt(1)>(0, 1);
    test_2<unsigned _BitInt(1)>(1, 1);
  }
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif
}
