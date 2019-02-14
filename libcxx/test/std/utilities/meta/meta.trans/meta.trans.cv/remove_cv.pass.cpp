//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_cv

#include <type_traits>

#include "test_macros.h"

template <class T, class U>
void test_remove_cv_imp()
{
    static_assert((std::is_same<typename std::remove_cv<T>::type, U>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::remove_cv_t<T>, U>::value), "");
#endif
}

template <class T>
void test_remove_cv()
{
    test_remove_cv_imp<T, T>();
    test_remove_cv_imp<const T, T>();
    test_remove_cv_imp<volatile T, T>();
    test_remove_cv_imp<const volatile T, T>();
}

int main(int, char**)
{
    test_remove_cv<void>();
    test_remove_cv<int>();
    test_remove_cv<int[3]>();
    test_remove_cv<int&>();
    test_remove_cv<const int&>();
    test_remove_cv<int*>();
    test_remove_cv<const int*>();

  return 0;
}
