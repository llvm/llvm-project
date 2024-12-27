//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// template <class T, class Alloc = allocator<T> >
// class list
// {
// public:
//
//     // types:
//     typedef T value_type;
//     typedef Alloc allocator_type;
//     typedef typename allocator_type::reference reference;
//     typedef typename allocator_type::const_reference const_reference;
//     typedef typename allocator_type::pointer pointer;
//     typedef typename allocator_type::const_pointer const_pointer;

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <list>
#include <type_traits>

#include "test_macros.h"
#include "min_allocator.h"

// Ensures that we don't use a non-uglified name 'base' in the implementation of 'list'.

struct my_base {
  typedef my_base base;
};

template <class T, class A = std::allocator<T> >
struct my_derived : my_base, std::list<T, A> {};

static_assert(std::is_same<my_derived<char>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<int>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<my_base>::base, my_base>::value, "");
#if TEST_STD_VER >= 11
static_assert(std::is_same<my_derived<char, min_allocator<char>>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<int, min_allocator<int>>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<my_base, min_allocator<my_base>>::base, my_base>::value, "");
#endif

struct A { std::list<A> v; }; // incomplete type support

int main(int, char**)
{
    {
    typedef std::list<int> C;
    static_assert((std::is_same<C::value_type, int>::value), "");
    static_assert((std::is_same<C::allocator_type, std::allocator<int> >::value), "");
    static_assert((std::is_same<C::reference, std::allocator_traits<std::allocator<int> >::value_type&>::value), "");
    static_assert(
        (std::is_same<C::const_reference, const std::allocator_traits<std::allocator<int> >::value_type&>::value), "");
    static_assert((std::is_same<C::pointer, std::allocator_traits<std::allocator<int> >::pointer>::value), "");
    static_assert(
        (std::is_same<C::const_pointer, std::allocator_traits<std::allocator<int> >::const_pointer>::value), "");

    static_assert((std::is_signed<typename C::difference_type>::value), "");
    static_assert((std::is_unsigned<typename C::size_type>::value), "");
    static_assert((std::is_same<typename C::difference_type,
        typename std::iterator_traits<typename C::iterator>::difference_type>::value), "");
    static_assert((std::is_same<typename C::difference_type,
        typename std::iterator_traits<typename C::const_iterator>::difference_type>::value), "");
    }

#if TEST_STD_VER >= 11
    {
    typedef std::list<int, min_allocator<int>> C;
    static_assert((std::is_same<C::value_type, int>::value), "");
    static_assert((std::is_same<C::allocator_type, min_allocator<int> >::value), "");
    static_assert((std::is_same<C::reference, int&>::value), "");
    static_assert((std::is_same<C::const_reference, const int&>::value), "");
    static_assert((std::is_same<C::pointer, min_pointer<int>>::value), "");
    static_assert((std::is_same<C::const_pointer, min_pointer<const int>>::value), "");

    static_assert((std::is_signed<typename C::difference_type>::value), "");
    static_assert((std::is_unsigned<typename C::size_type>::value), "");
    static_assert((std::is_same<typename C::difference_type,
        typename std::iterator_traits<typename C::iterator>::difference_type>::value), "");
    static_assert((std::is_same<typename C::difference_type,
        typename std::iterator_traits<typename C::const_iterator>::difference_type>::value), "");
    }
#endif

  return 0;
}
