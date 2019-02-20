//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_reference

#include <type_traits>
#include "test_macros.h"

template <class T, class U>
void test_remove_reference()
{
    static_assert((std::is_same<typename std::remove_reference<T>::type, U>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::remove_reference_t<T>, U>::value), "");
#endif
}

int main(int, char**)
{
    test_remove_reference<void, void>();
    test_remove_reference<int, int>();
    test_remove_reference<int[3], int[3]>();
    test_remove_reference<int*, int*>();
    test_remove_reference<const int*, const int*>();

    test_remove_reference<int&, int>();
    test_remove_reference<const int&, const int>();
    test_remove_reference<int(&)[3], int[3]>();
    test_remove_reference<int*&, int*>();
    test_remove_reference<const int*&, const int*>();

#if TEST_STD_VER >= 11
    test_remove_reference<int&&, int>();
    test_remove_reference<const int&&, const int>();
    test_remove_reference<int(&&)[3], int[3]>();
    test_remove_reference<int*&&, int*>();
    test_remove_reference<const int*&&, const int*>();
#endif

  return 0;
}
