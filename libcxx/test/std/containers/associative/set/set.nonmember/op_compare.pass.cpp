//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// template<class Key, class Compare, class Alloc>
// constexpr bool operator==(const std::set<Key, Compare, Alloc>& lhs,
//                 const std::set<Key, Compare, Alloc>& rhs); // constexpr since C++26
//
// template<class Key, class Compare, class Alloc>
// constexpr bool operator!=(const std::set<Key, Compare, Alloc>& lhs,
//                 const std::set<Key, Compare, Alloc>& rhs); // constexpr since C++26
//
// template<class Key, class Compare, class Alloc>
// constexpr bool operator<(const std::set<Key, Compare, Alloc>& lhs,
//                const std::set<Key, Compare, Alloc>& rhs); // constexpr since C++26
//
// template<class Key, class Compare, class Alloc>
// constexpr bool operator>(const std::set<Key, Compare, Alloc>& lhs,
//                const std::set<Key, Compare, Alloc>& rhs); // constexpr since C++26
//
// template<class Key, class Compare, class Alloc>
// constexpr bool operator<=(const std::set<Key, Compare, Alloc>& lhs,
//                 const std::set<Key, Compare, Alloc>& rhs); // constexpr since C++26
//
// template<class Key, class Compare, class Alloc>
// constexpr bool operator>=(const std::set<Key, Compare, Alloc>& lhs,
//                 const std::set<Key, Compare, Alloc>& rhs); // constexpr since C++26

#include <set>
#include <cassert>
#include <string>

#include "test_comparisons.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    std::set<int> s1, s2;
    s1.insert(1);
    s2.insert(2);
    const std::set<int>&cs1 = s1, cs2 = s2;
    assert(testComparisons(cs1, cs2, false, true));
  }
  {
    std::set<int> s1, s2;
    s1.insert(1);
    s2.insert(1);
    const std::set<int>&cs1 = s1, cs2 = s2;
    assert(testComparisons(cs1, cs2, true, false));
  }
  {
    std::set<int> s1, s2;
    s1.insert(1);
    s2.insert(1);
    s2.insert(2);
    const std::set<int>&cs1 = s1, cs2 = s2;
    assert(testComparisons(cs1, cs2, false, true));
  }
  return true;
}
int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
