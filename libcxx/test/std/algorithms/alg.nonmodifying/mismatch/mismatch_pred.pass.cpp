//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2,
//          Predicate<auto, Iter1::value_type, Iter2::value_type> Pred>
//   requires CopyConstructible<Pred>
//   pair<Iter1, Iter2>
//   mismatch(Iter1 first1, Iter1 last1, Iter2 first2, Pred pred);

#include <algorithm>
#include <functional>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "counting_predicates.hpp"

#if TEST_STD_VER > 11
#define HAS_FOUR_ITERATOR_VERSION
#endif

int main()
{
    int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    int ib[] = {0, 1, 2, 3, 0, 1, 2, 3};
    const unsigned sb = sizeof(ib)/sizeof(ib[0]); ((void)sb); // unused in C++11

    typedef input_iterator<const int*> II;
    typedef random_access_iterator<const int*>  RAI;
    typedef std::equal_to<int> EQ;

    assert(std::mismatch(II(ia), II(ia + sa), II(ib), EQ())
            == (std::pair<II, II>(II(ia+3), II(ib+3))));
    assert(std::mismatch(RAI(ia), RAI(ia + sa), RAI(ib), EQ())
            == (std::pair<RAI, RAI>(RAI(ia+3), RAI(ib+3))));

    binary_counting_predicate<EQ, int> bcp((EQ()));
    assert(std::mismatch(RAI(ia), RAI(ia + sa), RAI(ib), std::ref(bcp))
            == (std::pair<RAI, RAI>(RAI(ia+3), RAI(ib+3))));
    assert(bcp.count() > 0 && bcp.count() < sa);
    bcp.reset();

#if TEST_STD_VER >= 14
    assert(std::mismatch(II(ia), II(ia + sa), II(ib), II(ib + sb), EQ())
            == (std::pair<II, II>(II(ia+3), II(ib+3))));
    assert(std::mismatch(RAI(ia), RAI(ia + sa), RAI(ib), RAI(ib + sb), EQ())
            == (std::pair<RAI, RAI>(RAI(ia+3), RAI(ib+3))));

    assert(std::mismatch(II(ia), II(ia + sa), II(ib), II(ib + sb), std::ref(bcp))
            == (std::pair<II, II>(II(ia+3), II(ib+3))));
    assert(bcp.count() > 0 && bcp.count() < std::min(sa, sb));
#endif

    assert(std::mismatch(ia, ia + sa, ib, EQ()) ==
           (std::pair<int*,int*>(ia+3,ib+3)));

#if TEST_STD_VER >= 14
    assert(std::mismatch(ia, ia + sa, ib, ib + sb, EQ()) ==
           (std::pair<int*,int*>(ia+3,ib+3)));
    assert(std::mismatch(ia, ia + sa, ib, ib + 2, EQ()) ==
           (std::pair<int*,int*>(ia+2,ib+2)));
#endif
}
