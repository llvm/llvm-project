//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// type_traits

// remove_cvref

#include <type_traits>

#include "test_macros.h"

template <class T, class U>
void test_remove_cvref()
{
    ASSERT_SAME_TYPE(U, typename std::remove_cvref<T>::type);
    ASSERT_SAME_TYPE(U,        std::remove_cvref_t<T>);
}

int main(int, char**)
{
    test_remove_cvref<void, void>();
    test_remove_cvref<int, int>();
    test_remove_cvref<const int, int>();
    test_remove_cvref<const volatile int, int>();
    test_remove_cvref<volatile int, int>();

// Doesn't decay
    test_remove_cvref<int[3],                 int[3]>();
    test_remove_cvref<int const [3],          int[3]>();
    test_remove_cvref<int volatile [3],       int[3]>();
    test_remove_cvref<int const volatile [3], int[3]>();
    test_remove_cvref<void(), void ()>();

    test_remove_cvref<int &, int>();
    test_remove_cvref<const int &, int>();
    test_remove_cvref<const volatile int &, int>();
    test_remove_cvref<volatile int &, int>();

    test_remove_cvref<int*, int*>();
    test_remove_cvref<int(int) const, int(int) const>();
    test_remove_cvref<int(int) volatile, int(int) volatile>();
    test_remove_cvref<int(int)  &, int(int)  &>();
    test_remove_cvref<int(int) &&, int(int) &&>();

  return 0;
}
