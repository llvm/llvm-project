//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// template< class T, class Alloc >
// bool operator==( const std::list<T,Alloc>& lhs,
//                  const std::list<T,Alloc>& rhs );

// template< class T, class Alloc >
// bool operator!=( const std::list<T,Alloc>& lhs,
//                  const std::list<T,Alloc>& rhs );

// template< class T, class Alloc >
// bool operator<( const std::list<T,Alloc>& lhs,
//                 const std::list<T,Alloc>& rhs );

// template< class T, class Alloc >
// bool operator<=( const std::list<T,Alloc>& lhs,
//                  const std::list<T,Alloc>& rhs );

// template< class T, class Alloc >
// bool operator>( const std::list<T,Alloc>& lhs,
//                 const std::list<T,Alloc>& rhs );

// template< class T, class Alloc >
// bool operator>=( const std::list<T,Alloc>& lhs,
//                  const std::list<T,Alloc>& rhs );

#include <list>
#include <cassert>

#include "test_comparisons.h"

int main(int, char**) {
    {
        const std::list<int> l1, l2;
        assert(testComparisons(l1, l2, true, false));
    }
    {
        const std::list<int> l1(1, 1), l2(1, 1);
        assert(testComparisons(l1, l2, true, false));
    }
    {
        int items[3] = {1, 2, 3};
        const std::list<int> l1(items, items + 3);
        const std::list<int> l2(items, items + 3);
        assert(testComparisons(l1, l2, true, false));
    }
    {
        const std::list<int> l1(1, 1), l2;
        assert(testComparisons(l1, l2, false, false));
    }
    {
        const std::list<int> l1(1, 1), l2(1, 2);
        assert(testComparisons(l1, l2, false, true));
    }
    {
        int items1[2] = {1, 2};
        int items2[2] = {1, 3};
        const std::list<int> l1(items1, items1 + 2);
        const std::list<int> l2(items2, items2 + 2);
        assert(testComparisons(l1, l2, false, true));
    }
    {
        int items1[2] = {2, 2};
        int items2[2] = {1, 3};
        const std::list<int> l1(items1, items1 + 2);
        const std::list<int> l2(items2, items2 + 2);
        assert(testComparisons(l1, l2, false, false));
    }
    {
        const std::list<LessAndEqComp> l1, l2;
        assert(testComparisons(l1, l2, true, false));
    }
    {
        const std::list<LessAndEqComp> l1(1, LessAndEqComp(1));
        const std::list<LessAndEqComp> l2(1, LessAndEqComp(1));
        assert(testComparisons(l1, l2, true, false));
    }
    {
        LessAndEqComp items[3] = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(3)};
        const std::list<LessAndEqComp> l1(items, items + 3);
        const std::list<LessAndEqComp> l2(items, items + 3);
        assert(testComparisons(l1, l2, true, false));
    }
    {
        const std::list<LessAndEqComp> l1(1, LessAndEqComp(1));
        const std::list<LessAndEqComp> l2;
        assert(testComparisons(l1, l2, false, false));
    }
    {
        const std::list<LessAndEqComp> l1(1, LessAndEqComp(1));
        const std::list<LessAndEqComp> l2(1, LessAndEqComp(2));
        assert(testComparisons(l1, l2, false, true));
    }
    {
        LessAndEqComp items1[2] = {LessAndEqComp(1), LessAndEqComp(2)};
        LessAndEqComp items2[2] = {LessAndEqComp(1), LessAndEqComp(3)};
        const std::list<LessAndEqComp> l1(items1, items1 + 2);
        const std::list<LessAndEqComp> l2(items2, items2 + 2);
        assert(testComparisons(l1, l2, false, true));
    }
    {
        LessAndEqComp items1[2] = {LessAndEqComp(2), LessAndEqComp(2)};
        LessAndEqComp items2[2] = {LessAndEqComp(1), LessAndEqComp(3)};
        const std::list<LessAndEqComp> l1(items1, items1 + 2);
        const std::list<LessAndEqComp> l2(items2, items2 + 2);
        assert(testComparisons(l1, l2, false, false));
    }
    return 0;
}
