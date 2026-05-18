//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_signed

#include <type_traits>
#include <cstddef>

#include "test_macros.h"

template <class T>
void test_is_signed()
{
    static_assert( std::is_signed<T>::value, "");
    static_assert( std::is_signed<const T>::value, "");
    static_assert( std::is_signed<volatile T>::value, "");
    static_assert( std::is_signed<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_signed_v<T>, "");
    static_assert( std::is_signed_v<const T>, "");
    static_assert( std::is_signed_v<volatile T>, "");
    static_assert( std::is_signed_v<const volatile T>, "");
#endif
}

template <class T>
void test_is_not_signed()
{
    static_assert(!std::is_signed<T>::value, "");
    static_assert(!std::is_signed<const T>::value, "");
    static_assert(!std::is_signed<volatile T>::value, "");
    static_assert(!std::is_signed<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_signed_v<T>, "");
    static_assert(!std::is_signed_v<const T>, "");
    static_assert(!std::is_signed_v<volatile T>, "");
    static_assert(!std::is_signed_v<const volatile T>, "");
#endif
}

class Class
{
public:
    ~Class();
};

struct A; // incomplete

class incomplete_type;

class Empty {};

class NotEmpty {
  virtual ~NotEmpty();
};

union Union {};

struct bit_zero {
  int : 0;
};

class Abstract {
  virtual ~Abstract() = 0;
};

enum Enum { zero, one };

enum EnumSigned : int { two };

enum EnumUnsigned : unsigned { three };

enum class EnumClass { zero, one };

typedef void (*FunctionPtr)();

int main(int, char**)
{
  // Cases where !is_arithmetic implies !is_signed
  test_is_not_signed<std::nullptr_t>();
  test_is_not_signed<void>();
  test_is_not_signed<int&>();
  test_is_not_signed<int&&>();
  test_is_not_signed<Class>();
  test_is_not_signed<int*>();
  test_is_not_signed<const int*>();
  test_is_not_signed<char[3]>();
  test_is_not_signed<char[]>();
  test_is_not_signed<Union>();
  test_is_not_signed<Enum>();
  test_is_not_signed<EnumSigned>();
  test_is_not_signed<EnumUnsigned>();
  test_is_not_signed<EnumClass>();
  test_is_not_signed<FunctionPtr>();
  test_is_not_signed<Empty>();
  test_is_not_signed<incomplete_type>();
  test_is_not_signed<A>();
  test_is_not_signed<bit_zero>();
  test_is_not_signed<NotEmpty>();
  test_is_not_signed<Abstract>();

  test_is_signed<signed char>();
  test_is_signed<short>();
  test_is_signed<int>();
  test_is_signed<long>();
  test_is_signed<float>();
  test_is_signed<double>();

  test_is_not_signed<unsigned char>();
  test_is_not_signed<unsigned short>();
  test_is_not_signed<unsigned int>();
  test_is_not_signed<unsigned long>();

  test_is_not_signed<bool>();
  test_is_not_signed<unsigned>();

#ifndef TEST_HAS_NO_INT128
    test_is_signed<__int128_t>();
    test_is_not_signed<__uint128_t>();
#endif

  return 0;
}
