//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: Make this test pass for all standards.
// XFAIL: c++03

// <type_traits>

// __convert_to_integral(Tp)

// Test that the __convert_to_integral functions properly converts Tp to the
// correct type and value for integral, enum and user defined types.

#include <limits>
#include <type_traits>
#include <cstdint>
#include <cassert>

#include "user_defined_integral.h"

#include "test_macros.h"

template <class T>
struct EnumType
{
  enum type : T {E_zero, E_one};
};


template <class From, class To>
void check_integral_types()
{
  typedef std::numeric_limits<From> Limits;
  const From max = Limits::max();
  const From min = Limits::min();
  {
  auto ret = std::__convert_to_integral((From)max);
  assert(ret == max);
  ret = std::__convert_to_integral((From)min);
  assert(ret == min);
  static_assert(std::is_same<decltype(ret), To>::value, "");
  }
  {
  UserDefinedIntegral<From> f(max);
  auto ret = std::__convert_to_integral(f);
  assert(ret == max);
  f.value = min;
  ret = std::__convert_to_integral(f);
  assert(ret == min);
  static_assert(std::is_same<decltype(ret), To>::value, "");
  }
  {
  typedef typename EnumType<From>::type Enum;
  Enum e(static_cast<Enum>(max));
  auto ret = std::__convert_to_integral(e);
  assert(ret == max);
  e = static_cast<Enum>(min);
  ret = std::__convert_to_integral(min);
  assert(ret == min);
  static_assert(std::is_same<decltype(ret), To>::value, "");
  }
}


template <class From, class To>
void check_enum_types()
{
  auto ret = std::__convert_to_integral((From)1);
  assert(ret == 1);
  static_assert(std::is_same<decltype(ret), To>::value, "");
}


enum enum1 { zero = 0, one = 1 };
enum enum2 : unsigned long {
  value = std::numeric_limits<unsigned long>::max()
};

int main(int, char**)
{
  check_integral_types<bool, int>();
  check_integral_types<char, int>();
  check_integral_types<signed char, int>();
  check_integral_types<unsigned char, int>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  check_integral_types<wchar_t, decltype(((wchar_t)1) + 1)>();
#endif
  check_integral_types<char16_t, int>();
  // On some platforms, unsigned int and long are the same size.  These
  // platforms have a choice of making uint32_t an int or a long.  However
  // char32_t must promote to an unsigned int on these platforms [conv.prom].
  // Use the following logic to make the test work on such platforms.
  // (sizeof(uint32_t) == sizeof(unsigned int)) ? unsigned int : uint32_t;
  typedef std::conditional<sizeof(uint32_t) == sizeof(unsigned int),
                           unsigned int, uint32_t>::type char_integral;
  check_integral_types<char32_t, char_integral>();
  check_integral_types<short, int>();
  check_integral_types<unsigned short, int>();
  check_integral_types<int, int>();
  check_integral_types<unsigned, unsigned>();
  check_integral_types<long, long>();
  check_integral_types<unsigned long, unsigned long>();
  check_integral_types<long long, long long>();
  check_integral_types<unsigned long long, unsigned long long>();
#ifndef _LIBCPP_HAS_NO_INT128
  check_integral_types<__int128_t, __int128_t>();
  check_integral_types<__uint128_t, __uint128_t>();
#endif
    // TODO(ericwf): Not standard
  typedef std::underlying_type<enum1>::type Enum1UT;
  check_enum_types<enum1, decltype(((Enum1UT)1) + 1)>();
  typedef std::underlying_type<enum2>::type Enum2UT;
  check_enum_types<enum2, decltype(((Enum2UT)1) + 1)>();

  return 0;
}
