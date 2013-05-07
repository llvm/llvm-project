//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_reference

#include <type_traits>

template <class T, class U>
void test_remove_reference()
{
    static_assert((std::is_same<typename std::remove_reference<T>::type, U>::value), "");
}

int main()
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

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    test_remove_reference<int&&, int>();
    test_remove_reference<const int&&, const int>();
    test_remove_reference<int(&&)[3], int[3]>();
    test_remove_reference<int*&&, int*>();
    test_remove_reference<const int*&&, const int*>();
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
