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
// class back_insert_iterator {
// protected:
//   Cont* container;
// public:
//   typedef Cont                        container_type;
//   typedef void                        value_type;
//   typedef void                        difference_type;
//   typedef void                        reference;
//   typedef void                        pointer;
// };

#include <iterator>
#include <type_traits>
#include <vector>

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
