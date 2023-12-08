//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class Iter>
// struct iterator_traits
// {
//   typedef typename Iter::difference_type difference_type;
//   typedef typename Iter::value_type value_type;
//   typedef typename Iter::pointer pointer;
//   typedef typename Iter::reference reference;
//   typedef typename Iter::iterator_category iterator_category;
// };

#include <iterator>
#include <type_traits>

#include "test_macros.h"

struct A {};

struct test_iterator
{
    typedef int                       difference_type;
    typedef A                         value_type;
    typedef A*                        pointer;
    typedef A&                        reference;
    typedef std::forward_iterator_tag iterator_category;
};

int main(int, char**)
{
    typedef std::iterator_traits<test_iterator> It;
    static_assert((std::is_same<It::difference_type, int>::value), "");
    static_assert((std::is_same<It::value_type, A>::value), "");
    static_assert((std::is_same<It::pointer, A*>::value), "");
    static_assert((std::is_same<It::reference, A&>::value), "");
    static_assert((std::is_same<It::iterator_category, std::forward_iterator_tag>::value), "");

  return 0;
}
