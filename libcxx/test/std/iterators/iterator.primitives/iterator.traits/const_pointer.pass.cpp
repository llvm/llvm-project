//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class T>
// struct iterator_traits<const T*>
// {
//   typedef ptrdiff_t                  difference_type;
//   typedef T                          value_type;
//   typedef const T*                   pointer;
//   typedef const T&                   reference;
//   typedef random_access_iterator_tag iterator_category;
// };

#include <iterator>
#include <type_traits>

#include "test_macros.h"

struct A {};

int main(int, char**)
{
    typedef std::iterator_traits<const A*> It;
    static_assert((std::is_same<It::difference_type, std::ptrdiff_t>::value), "");
    static_assert((std::is_same<It::value_type, A>::value), "");
    static_assert((std::is_same<It::pointer, const A*>::value), "");
    static_assert((std::is_same<It::reference, const A&>::value), "");
    static_assert((std::is_same<It::iterator_category, std::random_access_iterator_tag>::value), "");

  return 0;
}
