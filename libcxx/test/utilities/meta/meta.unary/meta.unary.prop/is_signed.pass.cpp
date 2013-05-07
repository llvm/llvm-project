//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_signed

#include <type_traits>

template <class T>
void test_is_signed()
{
    static_assert( std::is_signed<T>::value, "");
    static_assert( std::is_signed<const T>::value, "");
    static_assert( std::is_signed<volatile T>::value, "");
    static_assert( std::is_signed<const volatile T>::value, "");
}

template <class T>
void test_is_not_signed()
{
    static_assert(!std::is_signed<T>::value, "");
    static_assert(!std::is_signed<const T>::value, "");
    static_assert(!std::is_signed<volatile T>::value, "");
    static_assert(!std::is_signed<const volatile T>::value, "");
}

class Class
{
public:
    ~Class();
};

int main()
{
    test_is_not_signed<void>();
    test_is_not_signed<int&>();
    test_is_not_signed<Class>();
    test_is_not_signed<int*>();
    test_is_not_signed<const int*>();
    test_is_not_signed<char[3]>();
    test_is_not_signed<char[3]>();
    test_is_not_signed<bool>();
    test_is_not_signed<unsigned>();

    test_is_signed<int>();
    test_is_signed<double>();
}
