//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2>
//   requires HasEqualTo<Iter1::value_type, Iter2::value_type>
//   bool
//   equal(Iter1 first1, Iter1 last1, Iter2 first2);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"


int main()
{
    int ia[] = {0, 1, 2, 3, 4, 5};
    const unsigned s = sizeof(ia)/sizeof(ia[0]);
    int ib[s] = {0, 1, 2, 5, 4, 5};
    assert(std::equal(input_iterator<const int*>(ia),
                      input_iterator<const int*>(ia+s),
                      input_iterator<const int*>(ia)));
#if TEST_STD_VER >= 14
    assert(std::equal(input_iterator<const int*>(ia),
                      input_iterator<const int*>(ia+s),
                      input_iterator<const int*>(ia),
                      input_iterator<const int*>(ia+s)));
    assert(std::equal(random_access_iterator<const int*>(ia),
                      random_access_iterator<const int*>(ia+s),
                      random_access_iterator<const int*>(ia),
                      random_access_iterator<const int*>(ia+s)));
#endif
    assert(!std::equal(input_iterator<const int*>(ia),
                       input_iterator<const int*>(ia+s),
                       input_iterator<const int*>(ib)));
#if TEST_STD_VER >= 14
    assert(!std::equal(input_iterator<const int*>(ia),
                       input_iterator<const int*>(ia+s),
                       input_iterator<const int*>(ib),
                       input_iterator<const int*>(ib+s)));
    assert(!std::equal(random_access_iterator<const int*>(ia),
                       random_access_iterator<const int*>(ia+s),
                       random_access_iterator<const int*>(ib),
                       random_access_iterator<const int*>(ib+s)));
    assert(!std::equal(input_iterator<const int*>(ia),
                       input_iterator<const int*>(ia+s),
                       input_iterator<const int*>(ia),
                       input_iterator<const int*>(ia+s-1)));
    assert(!std::equal(random_access_iterator<const int*>(ia),
                       random_access_iterator<const int*>(ia+s),
                       random_access_iterator<const int*>(ia),
                       random_access_iterator<const int*>(ia+s-1)));

#endif
}
