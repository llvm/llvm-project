//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_const

#include <type_traits>

template <class T>
void test_is_const()
{
    static_assert(!std::is_const<T>::value, "");
    static_assert( std::is_const<const T>::value, "");
    static_assert(!std::is_const<volatile T>::value, "");
    static_assert( std::is_const<const volatile T>::value, "");
}

int main()
{
    test_is_const<void>();
    test_is_const<int>();
    test_is_const<double>();
    test_is_const<int*>();
    test_is_const<const int*>();
    test_is_const<char[3]>();
    test_is_const<char[3]>();

    static_assert(!std::is_const<int&>::value, "");
    static_assert(!std::is_const<const int&>::value, "");
}
