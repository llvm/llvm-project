//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// template <class T, class Allocator = allocator<T>>
// class forward_list
// {
// public:
//   typedef T         value_type;
//   typedef Allocator allocator_type;
//
//   typedef value_type&                                                reference;
//   typedef const value_type&                                          const_reference;
//   typedef typename allocator_traits<allocator_type>::pointer         pointer;
//   typedef typename allocator_traits<allocator_type>::const_pointer   const_pointer;
//   typedef typename allocator_traits<allocator_type>::size_type       size_type;
//   typedef typename allocator_traits<allocator_type>::difference_type difference_type;
//   ...
// };

#include <forward_list>
#include <type_traits>

#include "test_macros.h"
#include "min_allocator.h"

// Ensures that we don't use a non-uglified name 'base' in the implementation of 'forward_list'.

struct my_base {
  typedef my_base base;
};

template <class T, class A = std::allocator<T> >
struct my_derived : my_base, std::forward_list<T, A> {};

static_assert(std::is_same<my_derived<char>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<int>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<my_base>::base, my_base>::value, "");
#if TEST_STD_VER >= 11
static_assert(std::is_same<my_derived<char, min_allocator<char>>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<int, min_allocator<int>>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<my_base, min_allocator<my_base>>::base, my_base>::value, "");
#endif

struct A {
  std::forward_list<A> v;
}; // incomplete type support

int main(int, char**) {
  {
    typedef std::forward_list<char> C;
    static_assert((std::is_same<C::value_type, char>::value), "");
    static_assert((std::is_same<C::allocator_type, std::allocator<char> >::value), "");
    static_assert((std::is_same<C::reference, char&>::value), "");
    static_assert((std::is_same<C::const_reference, const char&>::value), "");
    static_assert((std::is_same<C::pointer, char*>::value), "");
    static_assert((std::is_same<C::const_pointer, const char*>::value), "");
    static_assert((std::is_same<C::size_type, std::size_t>::value), "");
    static_assert((std::is_same<C::difference_type, std::ptrdiff_t>::value), "");

    static_assert((std::is_signed<typename C::difference_type>::value), "");
    static_assert((std::is_unsigned<typename C::size_type>::value), "");
    static_assert((std::is_same<typename C::difference_type,
                                typename std::iterator_traits<typename C::iterator>::difference_type>::value),
                  "");
    static_assert((std::is_same<typename C::difference_type,
                                typename std::iterator_traits<typename C::const_iterator>::difference_type>::value),
                  "");
  }
#if TEST_STD_VER >= 11
  {
    typedef std::forward_list<char, min_allocator<char>> C;
    static_assert((std::is_same<C::value_type, char>::value), "");
    static_assert((std::is_same<C::allocator_type, min_allocator<char> >::value), "");
    static_assert((std::is_same<C::reference, char&>::value), "");
    static_assert((std::is_same<C::const_reference, const char&>::value), "");
    static_assert((std::is_same<C::pointer, min_pointer<char>>::value), "");
    static_assert((std::is_same<C::const_pointer, min_pointer<const char>>::value), "");
    //  min_allocator doesn't have a size_type, so one gets synthesized
    static_assert((std::is_same<C::size_type, std::make_unsigned<C::difference_type>::type>::value), "");
    static_assert((std::is_same<C::difference_type, std::ptrdiff_t>::value), "");

    static_assert((std::is_signed<typename C::difference_type>::value), "");
    static_assert((std::is_unsigned<typename C::size_type>::value), "");
    static_assert((std::is_same<typename C::difference_type,
                                typename std::iterator_traits<typename C::iterator>::difference_type>::value),
                  "");
    static_assert((std::is_same<typename C::difference_type,
                                typename std::iterator_traits<typename C::const_iterator>::difference_type>::value),
                  "");
  }
#endif

  return 0;
}
