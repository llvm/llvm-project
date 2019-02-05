//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// void_t

// UNSUPPORTED: c++98, c++03, c++11, c++14

// XFAIL: gcc-5.1, gcc-5.2

#include <type_traits>

template <class T>
void test1()
{
    static_assert( std::is_same<void, std::void_t<T>>::value, "");
    static_assert( std::is_same<void, std::void_t<const T>>::value, "");
    static_assert( std::is_same<void, std::void_t<volatile T>>::value, "");
    static_assert( std::is_same<void, std::void_t<const volatile T>>::value, "");
}

template <class T, class U>
void test2()
{
    static_assert( std::is_same<void, std::void_t<T, U>>::value, "");
    static_assert( std::is_same<void, std::void_t<const T, U>>::value, "");
    static_assert( std::is_same<void, std::void_t<volatile T, U>>::value, "");
    static_assert( std::is_same<void, std::void_t<const volatile T, U>>::value, "");

    static_assert( std::is_same<void, std::void_t<T, const U>>::value, "");
    static_assert( std::is_same<void, std::void_t<const T, const U>>::value, "");
    static_assert( std::is_same<void, std::void_t<volatile T, const U>>::value, "");
    static_assert( std::is_same<void, std::void_t<const volatile T, const U>>::value, "");
}

class Class
{
public:
    ~Class();
};

int main(int, char**)
{
    static_assert( std::is_same<void, std::void_t<>>::value, "");

    test1<void>();
    test1<int>();
    test1<double>();
    test1<int&>();
    test1<Class>();
    test1<Class[]>();
    test1<Class[5]>();

    test2<void, int>();
    test2<double, int>();
    test2<int&, int>();
    test2<Class&, bool>();
    test2<void *, int&>();

    static_assert( std::is_same<void, std::void_t<int, double const &, Class, volatile int[], void>>::value, "");

  return 0;
}
