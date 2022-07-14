//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// template<class Key, class Compare, class Alloc>
// bool operator==(const std::multiset<Key, Compare, Alloc>& lhs,
//                 const std::multiset<Key, Compare, Alloc>& rhs);
//
// template<class Key, class Compare, class Alloc>
// bool operator!=(const std::multiset<Key, Compare, Alloc>& lhs,
//                 const std::multiset<Key, Compare, Alloc>& rhs);
//
// template<class Key, class Compare, class Alloc>
// bool operator<(const std::multiset<Key, Compare, Alloc>& lhs,
//                const std::multiset<Key, Compare, Alloc>& rhs);
//
// template<class Key, class Compare, class Alloc>
// bool operator>(const std::multiset<Key, Compare, Alloc>& lhs,
//                const std::multiset<Key, Compare, Alloc>& rhs);
//
// template<class Key, class Compare, class Alloc>
// bool operator<=(const std::multiset<Key, Compare, Alloc>& lhs,
//                 const std::multiset<Key, Compare, Alloc>& rhs);
//
// template<class Key, class Compare, class Alloc>
// bool operator>=(const std::multiset<Key, Compare, Alloc>& lhs,
//                 const std::multiset<Key, Compare, Alloc>& rhs);

#include <set>
#include <cassert>
#include <string>

#include "test_comparisons.h"

int main(int, char**) {
    {
        std::multiset<int> s1, s2;
        s1.insert(1);
        s2.insert(2);
        const std::multiset<int>& cs1 = s1, cs2 = s2;
        assert(testComparisons(cs1, cs2, false, true));
    }
    {
        std::multiset<int> s1, s2;
        s1.insert(1);
        s2.insert(1);
        const std::multiset<int>& cs1 = s1, cs2 = s2;
        assert(testComparisons(cs1, cs2, true, false));
    }
    {
        std::multiset<int> s1, s2;
        s1.insert(1);
        s2.insert(1);
        s2.insert(2);
        const std::multiset<int>& cs1 = s1, cs2 = s2;
        assert(testComparisons(cs1, cs2, false, true));
    }
    {
        std::multiset<int> s1, s2;
        s1.insert(1);
        s2.insert(1);
        s2.insert(1);
        s2.insert(1);
        const std::multiset<int>& cs1 = s1, cs2 = s2;
        assert(testComparisons(cs1, cs2, false, true));
    }
    return 0;
}
