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
#include "test_allocator.h"
#include "../../Copyable.h"
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

template <class C>
void test_iterators()
{
    typedef std::iterator_traits<typename C::iterator> ItT;
    typedef std::iterator_traits<typename C::const_iterator> CItT;

    static_assert((std::is_same<typename ItT::iterator_category, std::bidirectional_iterator_tag>::value), "");
    static_assert((std::is_same<typename ItT::value_type, typename C::value_type>::value), "");
    static_assert((std::is_same<typename ItT::reference, typename C::reference>::value), "");
    static_assert((std::is_same<typename ItT::pointer, typename C::pointer>::value), "");
    static_assert((std::is_same<typename ItT::difference_type, typename C::difference_type>::value), "");

    static_assert((std::is_same<typename CItT::iterator_category, std::bidirectional_iterator_tag>::value), "");
    static_assert((std::is_same<typename CItT::value_type, typename C::value_type>::value), "");
    static_assert((std::is_same<typename CItT::reference, typename C::const_reference>::value), "");
    static_assert((std::is_same<typename CItT::pointer, typename C::const_pointer>::value), "");
    static_assert((std::is_same<typename CItT::difference_type, typename C::difference_type>::value), "");
}

template <class T, class Allocator>
void test()
{
    typedef std::list<T, Allocator> C;
	typedef std::allocator_traits<Allocator> alloc_traits_t;

    static_assert((std::is_same<typename C::value_type, T>::value), "");
    static_assert((std::is_same<typename C::allocator_type, Allocator>::value), "");
    static_assert((std::is_same<typename C::reference, T&>::value), "");
    static_assert((std::is_same<typename C::const_reference, const T&>::value), "");
    static_assert((std::is_same<typename C::pointer, typename alloc_traits_t::pointer>::value), "");
    static_assert((std::is_same<typename C::const_pointer, typename alloc_traits_t::const_pointer>::value), "");
    static_assert((std::is_same<typename C::reverse_iterator, std::reverse_iterator<typename C::iterator> >::value), "");
    static_assert((std::is_same<typename C::const_reverse_iterator, std::reverse_iterator<typename C::const_iterator> >::value), "");

    static_assert((std::is_signed<typename C::difference_type>::value), "");
    static_assert((std::is_unsigned<typename C::size_type>::value), "");

	test_iterators<C>();
}

int main(int, char**)
{
	test<double, test_allocator<double>>();
	test<int*, test_allocator<int*>>();
	test<Copyable, test_allocator<Copyable>>();

#if TEST_STD_VER >= 11
	test<double, min_allocator<double>>();
    test<int*, min_allocator<int*>>();
    test<Copyable, min_allocator<Copyable>>();
#endif

  return 0;
}