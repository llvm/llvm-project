//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter, class T>
//   requires HasEqualTo<Iter::value_type, T>
//   Iter::difference_type
//   count(Iter first, Iter last, const T& value);

#include <algorithm>
#include <cassert>

#include "test_iterators.h"

int main()
{
    int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    assert(std::count(input_iterator<const int*>(ia),
                      input_iterator<const int*>(ia + sa), 2) == 3);
    assert(std::count(input_iterator<const int*>(ia),
                      input_iterator<const int*>(ia + sa), 7) == 0);
    assert(std::count(input_iterator<const int*>(ia),
                      input_iterator<const int*>(ia), 2) == 0);
}
