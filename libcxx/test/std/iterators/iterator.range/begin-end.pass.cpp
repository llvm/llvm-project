//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>
// template <class C> auto begin(C& c) -> decltype(c.begin());
// template <class C> auto begin(const C& c) -> decltype(c.begin());
// template <class C> auto end(C& c) -> decltype(c.end());
// template <class C> auto end(const C& c) -> decltype(c.end());
// template <class E> reverse_iterator<const E*> rbegin(initializer_list<E> il);
// template <class E> reverse_iterator<const E*> rend(initializer_list<E> il);

#include "test_macros.h"

#if TEST_STD_VER >= 11
#include <iterator>
#include <cassert>
#include <vector>
#include <array>
#include <list>
#include <initializer_list>

template<typename C>
void test_const_container( const C & c, typename C::value_type val ) {
    assert ( std::begin(c)   == c.begin());
    assert (*std::begin(c)   ==  val );
    assert ( std::begin(c)   != c.end());
    assert ( std::end(c)     == c.end());
#if _LIBCPP_STD_VER > 11
    assert ( std::cbegin(c)  == c.cbegin());
    assert ( std::cbegin(c)  != c.cend());
    assert ( std::cend(c)    == c.cend());
    assert ( std::rbegin(c)  == c.rbegin());
    assert ( std::rbegin(c)  != c.rend());
    assert ( std::rend(c)    == c.rend());
    assert ( std::crbegin(c) == c.crbegin());
    assert ( std::crbegin(c) != c.crend());
    assert ( std::crend(c)   == c.crend());
#endif
    }

template<typename T>
void test_const_container( const std::initializer_list<T> & c, T val ) {
    assert ( std::begin(c)   == c.begin());
    assert (*std::begin(c)   ==  val );
    assert ( std::begin(c)   != c.end());
    assert ( std::end(c)     == c.end());
#if _LIBCPP_STD_VER > 11
//  initializer_list doesn't have cbegin/cend/rbegin/rend
//  but std::cbegin(),etc work (b/c they're general fn templates)
//     assert ( std::cbegin(c)  == c.cbegin());
//     assert ( std::cbegin(c)  != c.cend());
//     assert ( std::cend(c)    == c.cend());
//     assert ( std::rbegin(c)  == c.rbegin());
//     assert ( std::rbegin(c)  != c.rend());
//     assert ( std::rend(c)    == c.rend());
//     assert ( std::crbegin(c) == c.crbegin());
//     assert ( std::crbegin(c) != c.crend());
//     assert ( std::crend(c)   == c.crend());
#endif
    }

template<typename C>
void test_container( C & c, typename C::value_type val ) {
    assert ( std::begin(c)   == c.begin());
    assert (*std::begin(c)   ==  val );
    assert ( std::begin(c)   != c.end());
    assert ( std::end(c)     == c.end());
#if _LIBCPP_STD_VER > 11
    assert ( std::cbegin(c)  == c.cbegin());
    assert ( std::cbegin(c)  != c.cend());
    assert ( std::cend(c)    == c.cend());
    assert ( std::rbegin(c)  == c.rbegin());
    assert ( std::rbegin(c)  != c.rend());
    assert ( std::rend(c)    == c.rend());
    assert ( std::crbegin(c) == c.crbegin());
    assert ( std::crbegin(c) != c.crend());
    assert ( std::crend(c)   == c.crend());
#endif
    }

template<typename T>
void test_container( std::initializer_list<T> & c, T val ) {
    assert ( std::begin(c)   == c.begin());
    assert (*std::begin(c)   ==  val );
    assert ( std::begin(c)   != c.end());
    assert ( std::end(c)     == c.end());
#if _LIBCPP_STD_VER > 11
//  initializer_list doesn't have cbegin/cend/rbegin/rend
//     assert ( std::cbegin(c)  == c.cbegin());
//     assert ( std::cbegin(c)  != c.cend());
//     assert ( std::cend(c)    == c.cend());
//     assert ( std::rbegin(c)  == c.rbegin());
//     assert ( std::rbegin(c)  != c.rend());
//     assert ( std::rend(c)    == c.rend());
//     assert ( std::crbegin(c) == c.crbegin());
//     assert ( std::crbegin(c) != c.crend());
//     assert ( std::crend(c)   == c.crend());
#endif
    }

template<typename T, size_t Sz>
void test_const_array( const T (&array)[Sz] ) {
    assert ( std::begin(array)  == array );
    assert (*std::begin(array)  ==  array[0] );
    assert ( std::begin(array)  != std::end(array));
    assert ( std::end(array)    == array + Sz);
#if _LIBCPP_STD_VER > 11
    assert ( std::cbegin(array) == array );
    assert (*std::cbegin(array) == array[0] );
    assert ( std::cbegin(array) != std::cend(array));
    assert ( std::cend(array)   == array + Sz);
#endif
    }

int main(){
    std::vector<int> v; v.push_back(1);
    std::list<int> l;   l.push_back(2);
    std::array<int, 1> a; a[0] = 3;
    std::initializer_list<int> il = { 4 };

    test_container ( v, 1 );
    test_container ( l, 2 );
    test_container ( a, 3 );
    test_container ( il, 4 );

    test_const_container ( v, 1 );
    test_const_container ( l, 2 );
    test_const_container ( a, 3 );
    test_const_container ( il, 4 );

    static constexpr int arrA [] { 1, 2, 3 };
    test_const_array ( arrA );
#if _LIBCPP_STD_VER > 11
    constexpr const int *b = std::cbegin(arrA);
    constexpr const int *e = std::cend(arrA);
    static_assert(e - b == 3, "");
#endif
}

#else
int main(){}
#endif
