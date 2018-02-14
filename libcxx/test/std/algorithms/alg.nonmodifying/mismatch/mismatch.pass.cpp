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
//   pair<Iter1, Iter2>
//   mismatch(Iter1 first1, Iter1 last1, Iter2 first2);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"


int main()
{
    int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    int ib[] = {0, 1, 2, 3, 0, 1, 2, 3};
    const unsigned sb = sizeof(ib)/sizeof(ib[0]); ((void)sb); // unused in C++11

    typedef input_iterator<const int*> II;
    typedef random_access_iterator<const int*>  RAI;

    assert(std::mismatch(II(ia), II(ia + sa), II(ib))
            == (std::pair<II, II>(II(ia+3), II(ib+3))));

    assert(std::mismatch(RAI(ia), RAI(ia + sa), RAI(ib))
            == (std::pair<RAI, RAI>(RAI(ia+3), RAI(ib+3))));

#if TEST_STD_VER > 11 // We have the four iteration version
    assert(std::mismatch(II(ia), II(ia + sa), II(ib), II(ib+sb))
            == (std::pair<II, II>(II(ia+3), II(ib+3))));

    assert(std::mismatch(RAI(ia), RAI(ia + sa), RAI(ib), RAI(ib+sb))
            == (std::pair<RAI, RAI>(RAI(ia+3), RAI(ib+3))));


    assert(std::mismatch(II(ia), II(ia + sa), II(ib), II(ib+2))
            == (std::pair<II, II>(II(ia+2), II(ib+2))));
#endif
}
