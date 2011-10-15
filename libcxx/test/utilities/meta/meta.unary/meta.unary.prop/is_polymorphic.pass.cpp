//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_polymorphic

#include <type_traits>

template <class T>
void test_is_polymorphic()
{
    static_assert( std::is_polymorphic<T>::value, "");
    static_assert( std::is_polymorphic<const T>::value, "");
    static_assert( std::is_polymorphic<volatile T>::value, "");
    static_assert( std::is_polymorphic<const volatile T>::value, "");
}

template <class T>
void test_is_not_polymorphic()
{
    static_assert(!std::is_polymorphic<T>::value, "");
    static_assert(!std::is_polymorphic<const T>::value, "");
    static_assert(!std::is_polymorphic<volatile T>::value, "");
    static_assert(!std::is_polymorphic<const volatile T>::value, "");
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

class Abstract
{
    virtual ~Abstract() = 0;
};

int main()
{
    test_is_not_polymorphic<void>();
    test_is_not_polymorphic<int&>();
    test_is_not_polymorphic<int>();
    test_is_not_polymorphic<double>();
    test_is_not_polymorphic<int*>();
    test_is_not_polymorphic<const int*>();
    test_is_not_polymorphic<char[3]>();
    test_is_not_polymorphic<char[3]>();
    test_is_not_polymorphic<Union>();
    test_is_not_polymorphic<Empty>();
    test_is_not_polymorphic<bit_zero>();

    test_is_polymorphic<NotEmpty>();
    test_is_polymorphic<Abstract>();
}
