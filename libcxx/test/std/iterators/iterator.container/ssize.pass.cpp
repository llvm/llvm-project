//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <iterator>
// template <class C> constexpr auto ssize(const C& c)
//     -> common_type_t<ptrdiff_t, make_signed_t<decltype(c.size())>>;                    // C++20
// template <class T, ptrdiff_t> constexpr ptrdiff_t ssize(const T (&array)[N]) noexcept; // C++20

#include <iterator>
#include <cassert>
#include <vector>
#include <array>
#include <list>
#include <initializer_list>
#include <string_view>

#include "test_macros.h"


struct short_container {
    uint16_t size() const { return 60000; } // not noexcept
    };



template<typename C>
void test_container(C& c)
{
//  Can't say noexcept here because the container might not be
    static_assert( std::is_signed_v<decltype(std::ssize(c))>, "");
    assert ( std::ssize(c)   == static_cast<decltype(std::ssize(c))>(c.size()));
}

template<typename C>
void test_const_container(const C& c)
{
//  Can't say noexcept here because the container might not be
    static_assert( std::is_signed_v<decltype(std::ssize(c))>, "");
    assert ( std::ssize(c)   == static_cast<decltype(std::ssize(c))>(c.size()));
}

template<typename T>
void test_const_container(const std::initializer_list<T>& c)
{
    LIBCPP_ASSERT_NOEXCEPT(std::ssize(c)); // our std::ssize is conditionally noexcept
    static_assert( std::is_signed_v<decltype(std::ssize(c))>, "");
    assert ( std::ssize(c)   == static_cast<decltype(std::ssize(c))>(c.size()));
}

template<typename T>
void test_container(std::initializer_list<T>& c)
{
    LIBCPP_ASSERT_NOEXCEPT(std::ssize(c)); // our std::ssize is conditionally noexcept
    static_assert( std::is_signed_v<decltype(std::ssize(c))>, "");
    assert ( std::ssize(c)   == static_cast<decltype(std::ssize(c))>(c.size()));
}

template<typename T, size_t Sz>
void test_const_array(const T (&array)[Sz])
{
    ASSERT_NOEXCEPT(std::ssize(array));
    static_assert( std::is_signed_v<decltype(std::ssize(array))>, "");
    assert ( std::ssize(array) == Sz );
}

int main(int, char**)
{
    std::vector<int> v; v.push_back(1);
    std::list<int>   l; l.push_back(2);
    std::array<int, 1> a; a[0] = 3;
    std::initializer_list<int> il = { 4 };
    test_container ( v );
    ASSERT_SAME_TYPE(ptrdiff_t, decltype(std::ssize(v)));
    test_container ( l );
    ASSERT_SAME_TYPE(ptrdiff_t, decltype(std::ssize(l)));
    test_container ( a );
    ASSERT_SAME_TYPE(ptrdiff_t, decltype(std::ssize(a)));
    test_container ( il );
    ASSERT_SAME_TYPE(ptrdiff_t, decltype(std::ssize(il)));

    test_const_container ( v );
    test_const_container ( l );
    test_const_container ( a );
    test_const_container ( il );

    std::string_view sv{"ABC"};
    test_container ( sv );
    ASSERT_SAME_TYPE(ptrdiff_t, decltype(std::ssize(sv)));
    test_const_container ( sv );

    static constexpr int arrA [] { 1, 2, 3 };
    ASSERT_SAME_TYPE(ptrdiff_t, decltype(std::ssize(arrA)));
    static_assert( std::is_signed_v<decltype(std::ssize(arrA))>, "");
    test_const_array ( arrA );

//  From P1227R2:
//     Note that the code does not just return the std::make_signed variant of
//     the container's size() method, because it's conceivable that a container
//     might choose to represent its size as a uint16_t, supporting up to
//     65,535 elements, and it would be a disaster for std::ssize() to turn a
//     size of 60,000 into a size of -5,536.

    short_container sc;
//  is the return type signed? Is it big enough to hold 60K?
//  is the "signed version" of sc.size() too small?
    static_assert( std::is_signed_v<                      decltype(std::ssize(sc))>, "");
    static_assert( std::numeric_limits<                   decltype(std::ssize(sc))>::max()  > 60000, "");
    static_assert( std::numeric_limits<std::make_signed_t<decltype(std:: size(sc))>>::max() < 60000, "");
    assert (std::ssize(sc) == 60000);
    LIBCPP_ASSERT_NOT_NOEXCEPT(std::ssize(sc));
    
  return 0;
}
