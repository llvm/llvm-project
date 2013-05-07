//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter, Predicate<auto, Iter::value_type> Pred>
//   requires CopyConstructible<Pred>
//   Iter
//   find_if_not(Iter first, Iter last, Pred pred);

#include <algorithm>
#include <functional>
#include <cassert>

#include "test_iterators.h"

int main()
{
    int ia[] = {0, 1, 2, 3, 4, 5};
    const unsigned s = sizeof(ia)/sizeof(ia[0]);
    input_iterator<const int*> r = std::find_if_not(input_iterator<const int*>(ia),
                                                    input_iterator<const int*>(ia+s),
                                                    std::bind2nd(std::not_equal_to<int>(), 3));
    assert(*r == 3);
    r = std::find_if_not(input_iterator<const int*>(ia),
                         input_iterator<const int*>(ia+s),
                         std::bind2nd(std::not_equal_to<int>(), 10));
    assert(r == input_iterator<const int*>(ia+s));
}
