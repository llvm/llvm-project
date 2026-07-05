//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_standard_layout

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_is_standard_layout()
{
    static_assert( std::is_standard_layout<T>::value, "");
    static_assert( std::is_standard_layout<const T>::value, "");
    static_assert( std::is_standard_layout<volatile T>::value, "");
    static_assert( std::is_standard_layout<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_standard_layout_v<T>, "");
    static_assert( std::is_standard_layout_v<const T>, "");
    static_assert( std::is_standard_layout_v<volatile T>, "");
    static_assert( std::is_standard_layout_v<const volatile T>, "");
#endif
}

template <class T>
void test_is_not_standard_layout()
{
    static_assert(!std::is_standard_layout<T>::value, "");
    static_assert(!std::is_standard_layout<const T>::value, "");
    static_assert(!std::is_standard_layout<volatile T>::value, "");
    static_assert(!std::is_standard_layout<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_standard_layout_v<T>, "");
    static_assert(!std::is_standard_layout_v<const T>, "");
    static_assert(!std::is_standard_layout_v<volatile T>, "");
    static_assert(!std::is_standard_layout_v<const volatile T>, "");
#endif
}

template <class T1, class T2>
struct pair
{
    T1 first;
    T2 second;
};

int main(int, char**)
{
    test_is_standard_layout<int> ();
    test_is_standard_layout<int[3]> ();
    test_is_standard_layout<pair<int, double> > ();

    test_is_not_standard_layout<int&> ();

  return 0;
}
