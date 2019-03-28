//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// add_volatile

#include <type_traits>

#include "test_macros.h"

template <class T, class U>
void test_add_volatile_imp()
{
    ASSERT_SAME_TYPE(volatile U, typename std::add_volatile<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(volatile U,        std::add_volatile_t<T>);
#endif
}

template <class T>
void test_add_volatile()
{
    test_add_volatile_imp<T, volatile T>();
    test_add_volatile_imp<const T, const volatile T>();
    test_add_volatile_imp<volatile T, volatile T>();
    test_add_volatile_imp<const volatile T, const volatile T>();
}

int main(int, char**)
{
    test_add_volatile<void>();
    test_add_volatile<int>();
    test_add_volatile<int[3]>();
    test_add_volatile<int&>();
    test_add_volatile<const int&>();
    test_add_volatile<int*>();
    test_add_volatile<const int*>();

  return 0;
}
