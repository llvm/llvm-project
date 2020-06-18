//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// Test nested types and data member:

// template <BidirectionalIterator Iter>
// class reverse_iterator {
// protected:
//   Iter current;
// public:
//   iterator<typename iterator_traits<Iterator>::iterator_category,
//   typename iterator_traits<Iterator>::value_type,
//   typename iterator_traits<Iterator>::difference_type,
//   typename iterator_traits<Iterator>::pointer,
//   typename iterator_traits<Iterator>::reference> {
// };

#include <iterator>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
struct find_current
    : private std::reverse_iterator<It>
{
    void test() { (void)this->current; }
};

template <class It>
void
test()
{
    typedef std::reverse_iterator<It> R;
    typedef std::iterator_traits<It> T;
    find_current<It> q; q.test(); // Just test that we can access `.current` from derived classes
    static_assert((std::is_same<typename R::iterator_type, It>::value), "");
    static_assert((std::is_same<typename R::value_type, typename T::value_type>::value), "");
    static_assert((std::is_same<typename R::difference_type, typename T::difference_type>::value), "");
    static_assert((std::is_same<typename R::reference, typename T::reference>::value), "");
    static_assert((std::is_same<typename R::pointer, typename std::iterator_traits<It>::pointer>::value), "");
    static_assert((std::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
}

int main(int, char**)
{
    test<bidirectional_iterator<char*> >();
    test<random_access_iterator<char*> >();
    test<char*>();

    return 0;
}
