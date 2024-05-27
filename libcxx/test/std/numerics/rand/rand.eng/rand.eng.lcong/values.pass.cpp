//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template <class UIntType, UIntType a, UIntType c, UIntType m>
// class linear_congruential_engine
// {
// public:
//     engine characteristics
//     static constexpr result_type multiplier = a;
//     static constexpr result_type increment = c;
//     static constexpr result_type modulus = m;
//     static constexpr result_type min() { return c == 0u ? 1u: 0u;}
//     static constexpr result_type max() { return m - 1u;}
//     static constexpr result_type default_seed = 1u;

#include <random>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <class T>
void where(const T &) {}

template <class T, T a, T c, T m>
void
test1()
{
    typedef std::linear_congruential_engine<T, a, c, m> LCE;
    typedef typename LCE::result_type result_type;
    static_assert((LCE::multiplier == a), "");
    static_assert((LCE::increment == c), "");
    static_assert((LCE::modulus == m), "");
#if TEST_STD_VER >= 11
    static_assert((LCE::min() == (c == 0u ? 1u: 0u)), "");
#else
    assert((LCE::min() == (c == 0u ? 1u: 0u)));
#endif

    TEST_DIAGNOSTIC_PUSH
    TEST_MSVC_DIAGNOSTIC_IGNORED(4310) // cast truncates constant value

#if TEST_STD_VER >= 11
    static_assert((LCE::max() == result_type(m - 1u)), "");
#else
    assert((LCE::max() == result_type(m - 1u)));
#endif

    TEST_DIAGNOSTIC_POP

    static_assert((LCE::default_seed == 1), "");
    where(LCE::multiplier);
    where(LCE::increment);
    where(LCE::modulus);
    where(LCE::default_seed);
}

template <class T>
void
test()
{
  const int W = sizeof(T) * CHAR_BIT;
  const T M(static_cast<T>(-1));
  const T A(static_cast<T>((static_cast<T>(1) << (W / 2)) - 1));

  // Cases where m = 0
  test1<T, 0, 0, 0>();
  test1<T, A, 0, 0>();
  test1<T, 0, 1, 0>();
  test1<T, A, 1, 0>();

  // Cases where m = 2^n for n < w
  test1<T, 0, 0, 256>();
  test1<T, 5, 0, 256>();
  test1<T, 0, 1, 256>();
  test1<T, 5, 1, 256>();

  // Cases where m is odd and a = 0
  test1<T, 0, 0, M>();
  test1<T, 0, M - 2, M>();
  test1<T, 0, M - 1, M>();

  // Cases where m is odd and m % a <= m / a (Schrage)
  test1<T, A, 0, M>();
  test1<T, A, M - 2, M>();
  test1<T, A, M - 1, M>();
}

template <class T>
void test_ext() {
  const T M(static_cast<T>(-1));

  // Cases where m is odd and m % a > m / a
  test1<T, M - 2, 0, M>();
  test1<T, M - 2, M - 2, M>();
  test1<T, M - 2, M - 1, M>();
  test1<T, M - 1, 0, M>();
  test1<T, M - 1, M - 2, M>();
  test1<T, M - 1, M - 1, M>();
}

int main(int, char**)
{
    test<unsigned short>();
    test_ext<unsigned short>();
    test<unsigned int>();
    test_ext<unsigned int>();
    test<unsigned long>();
    test_ext<unsigned long>();
    test<unsigned long long>();
    // This isn't implemented on platforms without __int128
#ifndef _LIBCPP_HAS_NO_INT128
    test_ext<unsigned long long>();
#endif

    return 0;
}
