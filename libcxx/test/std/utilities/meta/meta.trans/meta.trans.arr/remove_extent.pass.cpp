//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_extent

#include <type_traits>

#include "test_macros.h"

enum Enum {zero, one_};

template <class T, class U>
void test_remove_extent()
{
    ASSERT_SAME_TYPE(U, typename std::remove_extent<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(U,        std::remove_extent_t<T>);
#endif
}


int main(int, char**)
{
    test_remove_extent<int, int> ();
    test_remove_extent<const Enum, const Enum> ();
    test_remove_extent<int[], int> ();
    test_remove_extent<const int[], const int> ();
    test_remove_extent<int[3], int> ();
    test_remove_extent<const int[3], const int> ();
    test_remove_extent<int[][3], int[3]> ();
    test_remove_extent<const int[][3], const int[3]> ();
    test_remove_extent<int[2][3], int[3]> ();
    test_remove_extent<const int[2][3], const int[3]> ();
    test_remove_extent<int[1][2][3], int[2][3]> ();
    test_remove_extent<const int[1][2][3], const int[2][3]> ();

  return 0;
}
