//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// add_lvalue_reference

#include <type_traits>

template <class T, class U>
void test_add_lvalue_reference()
{
    static_assert((std::is_same<typename std::add_lvalue_reference<T>::type, U>::value), "");
}

int main()
{
    test_add_lvalue_reference<void, void>();
    test_add_lvalue_reference<int, int&>();
    test_add_lvalue_reference<int[3], int(&)[3]>();
    test_add_lvalue_reference<int&, int&>();
    test_add_lvalue_reference<const int&, const int&>();
    test_add_lvalue_reference<int*, int*&>();
    test_add_lvalue_reference<const int*, const int*&>();
}
