//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// back_insert_iterator

// Test nested types and data member:

// template <BackInsertionContainer Cont>
// class back_insert_iterator
//  : public iterator<output_iterator_tag, void, void, void, void> // until C++17
// {
// protected:
//   Cont* container;
// public:
//   typedef Cont                        container_type;
//   typedef void                        value_type;
//   typedef void                        difference_type; // until C++20
//   typedef ptrdiff_t                   difference_type; // since C++20
//   typedef void                        reference;
//   typedef void                        pointer;
// };

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

#include "test_macros.h"

template <class C>
struct find_container
    : private std::back_insert_iterator<C>
{
    explicit find_container(C& c) : std::back_insert_iterator<C>(c) {}
    void test() {this->container = 0;}
};

template <class C>
void
test()
{
    typedef std::back_insert_iterator<C> R;
    C c;
    find_container<C> q(c);
    q.test();
    static_assert((std::is_same<typename R::container_type, C>::value), "");
    static_assert((std::is_same<typename R::value_type, void>::value), "");
#if TEST_STD_VER > 17
    static_assert((std::is_same<typename R::difference_type, std::ptrdiff_t>::value), "");
#else
    static_assert((std::is_same<typename R::difference_type, void>::value), "");
#endif
    static_assert((std::is_same<typename R::reference, void>::value), "");
    static_assert((std::is_same<typename R::pointer, void>::value), "");
    static_assert((std::is_same<typename R::iterator_category, std::output_iterator_tag>::value), "");

#if TEST_STD_VER <= 14
    typedef std::iterator<std::output_iterator_tag, void, void, void, void> iterator_base;
    static_assert((std::is_base_of<iterator_base, R>::value), "");
#endif
}

int main(int, char**)
{
    test<std::vector<int> >();

  return 0;
}
