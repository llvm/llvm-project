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
//   Iter
//   find(Iter first, Iter last, const T& value);

#include <algorithm>
#include <cassert>

#include "../../iterators.h"

int main()
{
    int ia[] = {0, 1, 2, 3, 4, 5};
    const unsigned s = sizeof(ia)/sizeof(ia[0]);
    input_iterator<const int*> r = std::find(input_iterator<const int*>(ia),
                                             input_iterator<const int*>(ia+s), 3);
    assert(*r == 3);
    r = std::find(input_iterator<const int*>(ia), input_iterator<const int*>(ia+s), 10);
    assert(r == input_iterator<const int*>(ia+s));
}
