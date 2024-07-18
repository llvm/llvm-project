//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_pod

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_is_pod()
{
    static_assert( std::is_pod<T>::value, "");
    static_assert( std::is_pod<const T>::value, "");
    static_assert( std::is_pod<volatile T>::value, "");
    static_assert( std::is_pod<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_pod_v<T>, "");
    static_assert( std::is_pod_v<const T>, "");
    static_assert( std::is_pod_v<volatile T>, "");
    static_assert( std::is_pod_v<const volatile T>, "");
#endif
}

template <class T>
void test_is_not_pod()
{
    static_assert(!std::is_pod<T>::value, "");
    static_assert(!std::is_pod<const T>::value, "");
    static_assert(!std::is_pod<volatile T>::value, "");
    static_assert(!std::is_pod<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_pod_v<T>, "");
    static_assert(!std::is_pod_v<const T>, "");
    static_assert(!std::is_pod_v<volatile T>, "");
    static_assert(!std::is_pod_v<const volatile T>, "");
#endif
}

class Class
{
public:
    ~Class();
};

int main(int, char**)
{
    test_is_not_pod<void>();
    test_is_not_pod<int&>();
    test_is_not_pod<Class>();

    test_is_pod<int>();
    test_is_pod<double>();
    test_is_pod<int*>();
    test_is_pod<const int*>();
    test_is_pod<char[3]>();
    test_is_pod<char[]>();

  return 0;
}
