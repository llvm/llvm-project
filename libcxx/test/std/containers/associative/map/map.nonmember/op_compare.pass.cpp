//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// template<class Key, class T, class Compare, class Alloc>
// bool operator==(const std::map<Key, T, Compare, Alloc>& lhs,
//                 const std::map<Key, T, Compare, Alloc>& rhs); // constexpr since C++26
//
// template<class Key, class T, class Compare, class Alloc>
// bool operator!=(const std::map<Key, T, Compare, Alloc>& lhs,
//                 const std::map<Key, T, Compare, Alloc>& rhs); // constexpr since C++26
//
// template<class Key, class T, class Compare, class Alloc>
// bool operator<(const std::map<Key, T, Compare, Alloc>& lhs,
//                const std::map<Key, T, Compare, Alloc>& rhs); // constexpr since C++26
//
// template<class Key, class T, class Compare, class Alloc>
// bool operator>(const std::map<Key, T, Compare, Alloc>& lhs,
//                const std::map<Key, T, Compare, Alloc>& rhs); // constexpr since C++26
//
// template<class Key, class T, class Compare, class Alloc>
// bool operator<=(const std::map<Key, T, Compare, Alloc>& lhs,
//                 const std::map<Key, T, Compare, Alloc>& rhs); // constexpr since C++26
//
// template<class Key, class T, class Compare, class Alloc>
// bool operator>=(const std::map<Key, T, Compare, Alloc>& lhs,
//                 const std::map<Key, T, Compare, Alloc>& rhs); // constexpr since C++26

#include <map>
#include <cassert>
#include <string>

#include "test_comparisons.h"

TEST_CONSTEXPR_CXX26 bool test() {
  typedef std::map<int, std::string> map_type;
  typedef map_type::value_type value_type;
  {
    map_type m1, m2;
    m1.insert(value_type(1, "abc"));
    m2.insert(value_type(2, "abc"));
    const map_type &cm1 = m1, cm2 = m2;
    assert(testComparisons(cm1, cm2, false, true));
  }
  {
    map_type m1, m2;
    m1.insert(value_type(1, "abc"));
    m2.insert(value_type(1, "abc"));
    const map_type &cm1 = m1, cm2 = m2;
    assert(testComparisons(cm1, cm2, true, false));
  }
  {
    map_type m1, m2;
    m1.insert(value_type(1, "ab"));
    m2.insert(value_type(1, "abc"));
    const map_type &cm1 = m1, cm2 = m2;
    assert(testComparisons(cm1, cm2, false, true));
  }
  {
    map_type m1, m2;
    m1.insert(value_type(1, "abc"));
    m2.insert(value_type(1, "bcd"));
    const map_type &cm1 = m1, cm2 = m2;
    assert(testComparisons(cm1, cm2, false, true));
  }
  {
    map_type m1, m2;
    m1.insert(value_type(1, "abc"));
    m2.insert(value_type(1, "abc"));
    m2.insert(value_type(2, "abc"));
    const map_type &cm1 = m1, cm2 = m2;
    assert(testComparisons(cm1, cm2, false, true));
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
