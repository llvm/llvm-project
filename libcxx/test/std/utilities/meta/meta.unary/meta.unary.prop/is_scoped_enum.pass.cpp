//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// type_traits

// is_scoped_enum // C++2b

#include <type_traits>
#include <cstddef> // for std::nullptr_t
#include "test_macros.h"

template <class T>
void test_positive() {
  static_assert(std::is_scoped_enum<T>::value);
  static_assert(std::is_scoped_enum<const T>::value);
  static_assert(std::is_scoped_enum<volatile T>::value);
  static_assert(std::is_scoped_enum<const volatile T>::value);

  static_assert(std::is_scoped_enum_v<T>);
  static_assert(std::is_scoped_enum_v<const T>);
  static_assert(std::is_scoped_enum_v<volatile T>);
  static_assert(std::is_scoped_enum_v<const volatile T>);
}

template <class T>
void test_negative() {
  static_assert(!std::is_scoped_enum<T>::value);
  static_assert(!std::is_scoped_enum<const T>::value);
  static_assert(!std::is_scoped_enum<volatile T>::value);
  static_assert(!std::is_scoped_enum<const volatile T>::value);

  static_assert(!std::is_scoped_enum_v<T>);
  static_assert(!std::is_scoped_enum_v<const T>);
  static_assert(!std::is_scoped_enum_v<volatile T>);
  static_assert(!std::is_scoped_enum_v<const volatile T>);
}

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
enum class CEnum1 { zero, one };
enum class CEnum2;
enum class CEnum3 : short;
struct incomplete_type;

using FunctionPtr = void (*)();
using FunctionType = void();

struct TestMembers {
  static int static_method(int) { return 0; }
  int method() { return 0; }

  enum E1 { m_zero, m_one };
  enum class CE1;
};

void func1();
int func2(int);

int main(int, char**) {
  test_positive<CEnum1>();
  test_positive<CEnum2>();
  test_positive<CEnum3>();
  test_positive<TestMembers::CE1>();

  test_negative<Enum>();
  test_negative<TestMembers::E1>();

  test_negative<std::nullptr_t>();
  test_negative<void>();
  test_negative<int>();
  test_negative<int&>();
  test_negative<int&&>();
  test_negative<int*>();
  test_negative<double>();
  test_negative<const int*>();
  test_negative<char[3]>();
  test_negative<char[]>();
  test_negative<Union>();
  test_negative<Empty>();
  test_negative<bit_zero>();
  test_negative<NotEmpty>();
  test_negative<Abstract>();
  test_negative<FunctionPtr>();
  test_negative<FunctionType>();
  test_negative<incomplete_type>();
  test_negative<int TestMembers::*>();
  test_negative<void (TestMembers::*)()>();

  test_negative<decltype(func1)>();
  test_negative<decltype(&func1)>();
  test_negative<decltype(func2)>();
  test_negative<decltype(&func2)>();
  test_negative<decltype(TestMembers::static_method)>();
  test_negative<decltype(&TestMembers::static_method)>();
  test_negative<decltype(&TestMembers::method)>();

  return 0;
}
