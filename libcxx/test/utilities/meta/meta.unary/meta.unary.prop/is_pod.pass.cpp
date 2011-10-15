//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_pod

#include <type_traits>

template <class T>
void test_is_pod()
{
    static_assert( std::is_pod<T>::value, "");
    static_assert( std::is_pod<const T>::value, "");
    static_assert( std::is_pod<volatile T>::value, "");
    static_assert( std::is_pod<const volatile T>::value, "");
}

template <class T>
void test_is_not_pod()
{
    static_assert(!std::is_pod<T>::value, "");
    static_assert(!std::is_pod<const T>::value, "");
    static_assert(!std::is_pod<volatile T>::value, "");
    static_assert(!std::is_pod<const volatile T>::value, "");
}

class Class
{
public:
    ~Class();
};

int main()
{
    test_is_not_pod<void>();
    test_is_not_pod<int&>();
    test_is_not_pod<Class>();

    test_is_pod<int>();
    test_is_pod<double>();
    test_is_pod<int*>();
    test_is_pod<const int*>();
    test_is_pod<char[3]>();
    test_is_pod<char[3]>();
}
