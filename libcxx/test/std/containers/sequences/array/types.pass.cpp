//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// template <class T, size_t N >
// struct array
// {
//     // types:
//     typedef T& reference;
//     typedef const T& const_reference;
//     typedef implementation defined iterator;
//     typedef implementation defined const_iterator;
//     typedef T value_type;
//     typedef T* pointer;
//     typedef size_t size_type;
//     typedef ptrdiff_t difference_type;
//     typedef T value_type;
//     typedef std::reverse_iterator<iterator> reverse_iterator;
//     typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

#include <array>
#include <iterator>
#include <type_traits>

#include "../../Copyable.h"
#include "test_macros.h"

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

template <class T, std::size_t N>
void test()
{
        typedef std::array<T, N> C;

		static_assert((std::is_same<typename C::value_type, T>::value), "");
        static_assert((std::is_same<typename C::reference, T&>::value), "");
        static_assert((std::is_same<typename C::const_reference, const T&>::value), "");
        static_assert((std::is_same<typename C::pointer, T*>::value), "");
        static_assert((std::is_same<typename C::const_pointer, const T*>::value), "");
        static_assert((std::is_same<typename C::size_type, std::size_t>::value), "");
        static_assert((std::is_same<typename C::difference_type, std::ptrdiff_t>::value), "");
        static_assert((std::is_same<typename C::reverse_iterator, std::reverse_iterator<typename C::iterator> >::value), "");
        static_assert((std::is_same<typename C::const_reverse_iterator, std::reverse_iterator<typename C::const_iterator> >::value), "");

        static_assert((std::is_signed<typename C::difference_type>::value), "");
        static_assert((std::is_unsigned<typename C::size_type>::value), "");

        test_iterators<C>();
}

int main(int, char**)
{
    test<double, 10>();
    test<int*, 10>();
    test<Copyable, 10>();

    test<double, 0>();
    test<int*, 0>();
    test<Copyable, 0>();

  return 0;
}