//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// insert_iterator

// Test nested types and data members:

// template <InsertionContainer Cont>
// class insert_iterator {
// protected:
//   Cont* container;
//   Cont::iterator iter;
// public:
//   typedef Cont                   container_type;
//   typedef void                   value_type;
//   typedef void                   difference_type;
//   typedef void                   reference;
//   typedef void                   pointer;
// };

#include <iterator>
#include <type_traits>
#include <vector>
#include "test_macros.h"

template <class C>
struct find_members
    : private std::insert_iterator<C>
{
    explicit find_members(C& c) : std::insert_iterator<C>(c, c.begin()) {}
    void test()
    {
        this->container = 0;
        TEST_IGNORE_NODISCARD (this->iter == this->iter);
    }
};

template <class C>
void
test()
{
    typedef std::insert_iterator<C> R;
    C c;
    find_members<C> q(c);
    q.test();
    static_assert((std::is_same<typename R::container_type, C>::value), "");
    static_assert((std::is_same<typename R::value_type, void>::value), "");
    static_assert((std::is_same<typename R::difference_type, void>::value), "");
    static_assert((std::is_same<typename R::reference, void>::value), "");
    static_assert((std::is_same<typename R::pointer, void>::value), "");
    static_assert((std::is_same<typename R::iterator_category, std::output_iterator_tag>::value), "");
}

int main(int, char**)
{
    test<std::vector<int> >();

  return 0;
}
