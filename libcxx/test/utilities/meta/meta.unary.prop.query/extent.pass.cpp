//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// extent

#include <type_traits>

template <class T, unsigned A>
void test_extent()
{
    static_assert((std::extent<T>::value == A), "");
    static_assert((std::extent<const T>::value == A), "");
    static_assert((std::extent<volatile T>::value == A), "");
    static_assert((std::extent<const volatile T>::value == A), "");
}

template <class T, unsigned A>
void test_extent1()
{
    static_assert((std::extent<T, 1>::value == A), "");
    static_assert((std::extent<const T, 1>::value == A), "");
    static_assert((std::extent<volatile T, 1>::value == A), "");
    static_assert((std::extent<const volatile T, 1>::value == A), "");
}

class Class
{
public:
    ~Class();
};

int main()
{
    test_extent<void, 0>();
    test_extent<int&, 0>();
    test_extent<Class, 0>();
    test_extent<int*, 0>();
    test_extent<const int*, 0>();
    test_extent<int, 0>();
    test_extent<double, 0>();
    test_extent<bool, 0>();
    test_extent<unsigned, 0>();

    test_extent<int[2], 2>();
    test_extent<int[2][4], 2>();
    test_extent<int[][4], 0>();

    test_extent1<int, 0>();
    test_extent1<int[2], 0>();
    test_extent1<int[2][4], 4>();
    test_extent1<int[][4], 4>();
}
