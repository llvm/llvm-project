//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <iterator>
// template <class C> constexpr auto data(C& c) -> decltype(c.data());               // C++17
// template <class C> constexpr auto data(const C& c) -> decltype(c.data());         // C++17
// template <class T, size_t N> constexpr T* data(T (&array)[N]) noexcept;           // C++17

#include <array>
#include <cassert>
#include <iterator>
#include <initializer_list>
#include <string_view>
#include <type_traits>
#include <vector>

#include "test_macros.h"

template<typename C>
void test_const_container( const C& c )
{
  static_assert(noexcept(std::data(c)) == noexcept(c.data()));
  assert(std::data(c) == c.data());
}

template<typename T>
void test_const_container( const std::initializer_list<T>& c )
{
    ASSERT_NOEXCEPT(std::data(c));
    assert ( std::data(c)   == c.begin());
}

template<typename C>
void test_container( C& c )
{
  static_assert(noexcept(std::data(c)) == noexcept(c.data()));
  assert(std::data(c) == c.data());
}

template<typename T>
void test_container( std::initializer_list<T>& c)
{
    ASSERT_NOEXCEPT(std::data(c));
    assert ( std::data(c)   == c.begin());
}

template<typename T, std::size_t Sz>
void test_const_array( const T (&array)[Sz] )
{
    ASSERT_NOEXCEPT(std::data(array));
    assert ( std::data(array) == &array[0]);
}

// Verify that the std::data overload for std::initializer_list is removed by P3016R6.
template <class T, class = void>
constexpr bool can_std_data_with_int_list = false;
template <class T>
constexpr bool can_std_data_with_int_list<T, std::void_t<decltype(std::data<T>({1, 2, 3}))>> = true;

static_assert(!can_std_data_with_int_list<int>);
// When C is a range type convertible from {1, 2, 3}, the overload for const C& is selected.
static_assert(can_std_data_with_int_list<std::vector<int>>);
static_assert(can_std_data_with_int_list<std::array<int, 3>>);
static_assert(can_std_data_with_int_list<std::initializer_list<int>>);

int main(int, char**)
{
    std::vector<int> v; v.push_back(1);
    std::array<int, 1> a; a[0] = 3;
    std::initializer_list<int> il = { 4 };

    test_container ( v );
    test_container ( a );
    test_container ( il );

    test_const_container ( v );
    test_const_container ( a );
    test_const_container ( il );

    std::string_view sv{"ABC"};
    test_container(sv);
    test_const_container(sv);

    static constexpr int arrA [] { 1, 2, 3 };
    test_const_array ( arrA );

  return 0;
}
