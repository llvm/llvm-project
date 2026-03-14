//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// union

#include <type_traits>

#include "test_macros.h"

template <class T>
void test_union_imp()
{
    static_assert(!std::is_reference<T>::value, "");
    static_assert(!std::is_arithmetic<T>::value, "");
    static_assert(!std::is_fundamental<T>::value, "");
    static_assert( std::is_object<T>::value, "");
    static_assert(!std::is_scalar<T>::value, "");
    static_assert( std::is_compound<T>::value, "");
    static_assert(!std::is_member_pointer<T>::value, "");
}

template <class T>
void test_union()
{
    test_union_imp<T>();
    test_union_imp<const T>();
    test_union_imp<volatile T>();
    test_union_imp<const volatile T>();
}

union Union
{
    int _;
    double __;
};

int main(int, char**)
{
    test_union<Union>();

  return 0;
}
