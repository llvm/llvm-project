//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// member_object_pointer

#include <type_traits>

template <class T>
void test_member_object_pointer_imp()
{
    static_assert(!std::is_reference<T>::value, "");
    static_assert(!std::is_arithmetic<T>::value, "");
    static_assert(!std::is_fundamental<T>::value, "");
    static_assert( std::is_object<T>::value, "");
    static_assert( std::is_scalar<T>::value, "");
    static_assert( std::is_compound<T>::value, "");
    static_assert( std::is_member_pointer<T>::value, "");
}

template <class T>
void test_member_object_pointer()
{
    test_member_object_pointer_imp<T>();
    test_member_object_pointer_imp<const T>();
    test_member_object_pointer_imp<volatile T>();
    test_member_object_pointer_imp<const volatile T>();
}

class Class
{
};

int main()
{
    test_member_object_pointer<int Class::*>();
}
