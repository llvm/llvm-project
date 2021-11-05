//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// template<class Key, class T, class Compare, class Alloc>
// bool operator==(const std::multimap<Key, T, Compare, Alloc>& lhs,
//                 const std::multimap<Key, T, Compare, Alloc>& rhs);
//
// template<class Key, class T, class Compare, class Alloc>
// bool operator!=(const std::multimap<Key, T, Compare, Alloc>& lhs,
//                 const std::multimap<Key, T, Compare, Alloc>& rhs);
//
// template<class Key, class T, class Compare, class Alloc>
// bool operator<(const std::multimap<Key, T, Compare, Alloc>& lhs,
//                const std::multimap<Key, T, Compare, Alloc>& rhs);
//
// template<class Key, class T, class Compare, class Alloc>
// bool operator>(const std::multimap<Key, T, Compare, Alloc>& lhs,
//                const std::multimap<Key, T, Compare, Alloc>& rhs);
//
// template<class Key, class T, class Compare, class Alloc>
// bool operator<=(const std::multimap<Key, T, Compare, Alloc>& lhs,
//                 const std::multimap<Key, T, Compare, Alloc>& rhs);
//
// template<class Key, class T, class Compare, class Alloc>
// bool operator>=(const std::multimap<Key, T, Compare, Alloc>& lhs,
//                 const std::multimap<Key, T, Compare, Alloc>& rhs);

#include <map>
#include <cassert>
#include <string>

#include "test_comparisons.h"

int main(int, char**) {
    typedef std::multimap<int, std::string> map_type;
    typedef map_type::value_type value_type;
    {
        map_type m1, m2;
        m1.insert(value_type(1, "abc"));
        m2.insert(value_type(2, "abc"));
        const map_type& cm1 = m1, cm2 = m2;
        assert(testComparisons6(cm1, cm2, false, true));
    }
    {
        map_type m1, m2;
        m1.insert(value_type(1, "abc"));
        m2.insert(value_type(1, "abc"));
        const map_type& cm1 = m1, cm2 = m2;
        assert(testComparisons6(cm1, cm2, true, false));
    }
    {
        map_type m1, m2;
        m1.insert(value_type(1, "ab"));
        m2.insert(value_type(1, "abc"));
        const map_type& cm1 = m1, cm2 = m2;
        assert(testComparisons6(cm1, cm2, false, true));
    }
    {
        map_type m1, m2;
        m1.insert(value_type(1, "abc"));
        m2.insert(value_type(1, "bcd"));
        const map_type& cm1 = m1, cm2 = m2;
        assert(testComparisons6(cm1, cm2, false, true));
    }
    {
        map_type m1, m2;
        m1.insert(value_type(1, "abc"));
        m2.insert(value_type(1, "abc"));
        m2.insert(value_type(2, "abc"));
        const map_type& cm1 = m1, cm2 = m2;
        assert(testComparisons6(cm1, cm2, false, true));
    }
    {
        map_type m1, m2;
        m1.insert(value_type(1, "abc"));
        m2.insert(value_type(1, "abc"));
        m2.insert(value_type(1, "abc"));
        m2.insert(value_type(1, "bcd"));
        const map_type& cm1 = m1, cm2 = m2;
        assert(testComparisons6(cm1, cm2, false, true));
    }
    return 0;
}
