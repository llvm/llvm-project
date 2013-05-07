//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_empty

#include <type_traits>

template <class T>
void test_is_empty()
{
    static_assert( std::is_empty<T>::value, "");
    static_assert( std::is_empty<const T>::value, "");
    static_assert( std::is_empty<volatile T>::value, "");
    static_assert( std::is_empty<const volatile T>::value, "");
}

template <class T>
void test_is_not_empty()
{
    static_assert(!std::is_empty<T>::value, "");
    static_assert(!std::is_empty<const T>::value, "");
    static_assert(!std::is_empty<volatile T>::value, "");
    static_assert(!std::is_empty<const volatile T>::value, "");
}

class Empty
{
};

class NotEmpty
{
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

int main()
{
    test_is_not_empty<void>();
    test_is_not_empty<int&>();
    test_is_not_empty<int>();
    test_is_not_empty<double>();
    test_is_not_empty<int*>();
    test_is_not_empty<const int*>();
    test_is_not_empty<char[3]>();
    test_is_not_empty<char[3]>();
    test_is_not_empty<Union>();
    test_is_not_empty<NotEmpty>();

    test_is_empty<Empty>();
    test_is_empty<bit_zero>();
}
