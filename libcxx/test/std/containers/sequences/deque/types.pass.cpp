//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// Test nested types and default template args:

// template <class T, class Allocator = allocator<T> >
// class deque
// {
// public:
//     typedef T                                        value_type;
//     typedef Allocator                                allocator_type;
//     typedef typename allocator_type::reference       reference;
//     typedef typename allocator_type::const_reference const_reference;
//     typedef implementation-defined                   iterator;
//     typedef implementation-defined                   const_iterator;
//     typedef typename allocator_type::size_type       size_type;
//     typedef typename allocator_type::difference_type difference_type;
//     typedef typename allocator_type::pointer         pointer;
//     typedef typename allocator_type::const_pointer   const_pointer;
//     typedef std::reverse_iterator<iterator>          reverse_iterator;
//     typedef std::reverse_iterator<const_iterator>    const_reverse_iterator;
// };

#include <deque>
#include <iterator>
#include <type_traits>

#include "test_macros.h"
#include "test_allocator.h"
#include "../../Copyable.h"
#include "min_allocator.h"

template <class C>
void test_iterators()
{
    typedef std::iterator_traits<typename C::iterator> ItT;
    typedef std::iterator_traits<typename C::const_iterator> CItT;

    static_assert((std::is_same<typename ItT::iterator_category, std::random_access_iterator_tag>::value), "");
    static_assert((std::is_same<typename ItT::value_type, typename C::value_type>::value), "");
    static_assert((std::is_same<typename ItT::reference, typename C::reference>::value), "");
    static_assert((std::is_same<typename ItT::pointer, typename C::pointer>::value), "");
    static_assert((std::is_same<typename ItT::difference_type, typename C::difference_type>::value), "");

    static_assert((std::is_same<typename CItT::iterator_category, std::random_access_iterator_tag>::value), "");
    static_assert((std::is_same<typename CItT::value_type, typename C::value_type>::value), "");
    static_assert((std::is_same<typename CItT::reference, typename C::const_reference>::value), "");
    static_assert((std::is_same<typename CItT::pointer, typename C::const_pointer>::value), "");
    static_assert((std::is_same<typename CItT::difference_type, typename C::difference_type>::value), "");
}

template <class T, class Allocator>
void test()
{
    typedef std::deque<T, Allocator> C;
    typedef std::allocator_traits<Allocator> alloc_traits_t;

    static_assert((std::is_same<typename C::value_type, T>::value), "");
    static_assert((std::is_same<typename C::allocator_type, Allocator>::value), "");
    static_assert((std::is_same<typename C::size_type, typename alloc_traits_t::size_type>::value), "");
    static_assert((std::is_same<typename C::difference_type, typename alloc_traits_t::difference_type>::value), "");
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
