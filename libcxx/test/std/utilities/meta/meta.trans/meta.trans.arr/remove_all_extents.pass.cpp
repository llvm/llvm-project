//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_all_extents

#include <type_traits>

#include "test_macros.h"

enum Enum {zero, one_};

template <class T, class U>
void test_remove_all_extents()
{
    ASSERT_SAME_TYPE(U, typename std::remove_all_extents<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(U,        std::remove_all_extents_t<T>);
#endif
}

int main(int, char**)
{
    test_remove_all_extents<int, int> ();
    test_remove_all_extents<const Enum, const Enum> ();
    test_remove_all_extents<int[], int> ();
    test_remove_all_extents<const int[], const int> ();
    test_remove_all_extents<int[3], int> ();
    test_remove_all_extents<const int[3], const int> ();
    test_remove_all_extents<int[][3], int> ();
    test_remove_all_extents<const int[][3], const int> ();
    test_remove_all_extents<int[2][3], int> ();
    test_remove_all_extents<const int[2][3], const int> ();
    test_remove_all_extents<int[1][2][3], int> ();
    test_remove_all_extents<const int[1][2][3], const int> ();

  return 0;
}
