//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// add_cv

#include <type_traits>

#include "test_macros.h"

template <class T, class U>
void test_add_cv_imp()
{
    ASSERT_SAME_TYPE(const volatile U, typename std::add_cv<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(const volatile U,        std::add_cv_t<T>);
#endif
}

template <class T>
void test_add_cv()
{
    test_add_cv_imp<T, const volatile T>();
    test_add_cv_imp<const T, const volatile T>();
    test_add_cv_imp<volatile T, volatile const T>();
    test_add_cv_imp<const volatile T, const volatile T>();
}

int main(int, char**)
{
    test_add_cv<void>();
    test_add_cv<int>();
    test_add_cv<int[3]>();
    test_add_cv<int&>();
    test_add_cv<const int&>();
    test_add_cv<int*>();
    test_add_cv<const int*>();

  return 0;
}
