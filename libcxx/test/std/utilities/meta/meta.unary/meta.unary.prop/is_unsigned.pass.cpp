//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_unsigned

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_is_unsigned()
{
    static_assert( std::is_unsigned<T>::value, "");
    static_assert( std::is_unsigned<const T>::value, "");
    static_assert( std::is_unsigned<volatile T>::value, "");
    static_assert( std::is_unsigned<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_unsigned_v<T>, "");
    static_assert( std::is_unsigned_v<const T>, "");
    static_assert( std::is_unsigned_v<volatile T>, "");
    static_assert( std::is_unsigned_v<const volatile T>, "");
#endif
}

template <class T>
void test_is_not_unsigned()
{
    static_assert(!std::is_unsigned<T>::value, "");
    static_assert(!std::is_unsigned<const T>::value, "");
    static_assert(!std::is_unsigned<volatile T>::value, "");
    static_assert(!std::is_unsigned<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_unsigned_v<T>, "");
    static_assert(!std::is_unsigned_v<const T>, "");
    static_assert(!std::is_unsigned_v<volatile T>, "");
    static_assert(!std::is_unsigned_v<const volatile T>, "");
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
  // Cases where !is_arithmetic implies !is_unsigned
  test_is_not_unsigned<std::nullptr_t>();
  test_is_not_unsigned<void>();
  test_is_not_unsigned<int&>();
  test_is_not_unsigned<int&&>();
  test_is_not_unsigned<Class>();
  test_is_not_unsigned<int*>();
  test_is_not_unsigned<const int*>();
  test_is_not_unsigned<char[3]>();
  test_is_not_unsigned<char[]>();
  test_is_not_unsigned<Union>();
  test_is_not_unsigned<Enum>();
  test_is_not_unsigned<EnumSigned>();
  test_is_not_unsigned<EnumUnsigned>();
  test_is_not_unsigned<EnumClass>();
  test_is_not_unsigned<FunctionPtr>();
  test_is_not_unsigned<Empty>();
  test_is_not_unsigned<incomplete_type>();
  test_is_not_unsigned<A>();
  test_is_not_unsigned<bit_zero>();
  test_is_not_unsigned<NotEmpty>();
  test_is_not_unsigned<Abstract>();

  test_is_not_unsigned<signed char>();
  test_is_not_unsigned<short>();
  test_is_not_unsigned<int>();
  test_is_not_unsigned<long>();
  test_is_not_unsigned<float>();
  test_is_not_unsigned<double>();

  test_is_unsigned<unsigned char>();
  test_is_unsigned<unsigned short>();
  test_is_unsigned<unsigned int>();
  test_is_unsigned<unsigned long>();

  test_is_unsigned<bool>();
  test_is_unsigned<unsigned>();

#ifndef _LIBCPP_HAS_NO_INT128
    test_is_unsigned<__uint128_t>();
    test_is_not_unsigned<__int128_t>();
#endif

  return 0;
}
