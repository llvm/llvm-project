//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// Test nested types:

// template <InputIterator Iter>
// class move_iterator {
// public:
//   typedef Iter                  iterator_type;
//   typedef Iter::difference_type difference_type;
//   typedef Iter                  pointer;
//   typedef Iter::value_type      value_type;
//   typedef value_type&&          reference;
// };

#include <iterator>
#include <type_traits>

#include "test_iterators.h"

template <class It>
void
test()
{
    typedef std::move_iterator<It> R;
    typedef std::iterator_traits<It> T;
    static_assert((std::is_same<typename R::iterator_type, It>::value), "");
    static_assert((std::is_same<typename R::difference_type, typename T::difference_type>::value), "");
    static_assert((std::is_same<typename R::pointer, It>::value), "");
    static_assert((std::is_same<typename R::value_type, typename T::value_type>::value), "");
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    static_assert((std::is_same<typename R::reference, typename R::value_type&&>::value), "");
#else
    static_assert((std::is_same<typename R::reference, typename T::reference>::value), "");
#endif
    static_assert((std::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
}

int main()
{
    test<input_iterator<char*> >();
    test<forward_iterator<char*> >();
    test<bidirectional_iterator<char*> >();
    test<random_access_iterator<char*> >();
    test<char*>();
}
