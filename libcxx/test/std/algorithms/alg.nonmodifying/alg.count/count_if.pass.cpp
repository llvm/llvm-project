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
//   Iter::difference_type
//   count_if(Iter first, Iter last, Pred pred);

#include <algorithm>
#include <functional>
#include <cassert>

#include "test_iterators.h"

struct eq {
	eq (int val) : v(val) {}
	bool operator () (int v2) const { return v == v2; }
	int v;
	};
	

int main()
{
    int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    assert(std::count_if(input_iterator<const int*>(ia),
                         input_iterator<const int*>(ia + sa),
                         eq(2)) == 3);
    assert(std::count_if(input_iterator<const int*>(ia),
                         input_iterator<const int*>(ia + sa),
                         eq(7)) == 0);
    assert(std::count_if(input_iterator<const int*>(ia),
                         input_iterator<const int*>(ia),
                         eq(2)) == 0);
}
