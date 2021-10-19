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

#include <iterator>
#include <type_traits>

#include "test_macros.h"

struct A {};

int main(int, char**)
{
    typedef std::iterator_traits<volatile A*> It;
    static_assert((std::is_same<It::difference_type, std::ptrdiff_t>::value), "");
    static_assert((std::is_same<It::value_type, A>::value), "");
    static_assert((std::is_same<It::pointer, volatile A*>::value), "");
    static_assert((std::is_same<It::reference, volatile A&>::value), "");
    static_assert((std::is_same<It::iterator_category, std::random_access_iterator_tag>::value), "");
#if TEST_STD_VER > 17
    ASSERT_SAME_TYPE(It::iterator_concept, std::contiguous_iterator_tag);
#endif
    return 0;
}
