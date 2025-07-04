//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>
//
// flat_set should support comparator that return a non-boolean
// value as long as the returned type is implicitly convertible to bool.

#include <flat_set>
#include <vector>
#include <ranges>

#include "boolean_testable.h"

void test() {
  using Key = StrictComparable<int>;
  std::vector<Key> v;
  std::flat_set<Key> m1;
  std::flat_set m2(std::from_range, v, StrictBinaryPredicate);
  std::flat_set m3(std::sorted_unique, v, StrictBinaryPredicate);
  std::flat_set m4(m1.begin(), m1.end(), StrictBinaryPredicate);
  m2.insert(m1.begin(), m1.end());
  m2.insert(std::sorted_unique, m1.begin(), m1.end());
  m2.insert_range(m1);
  m3.insert(1);
  m2.emplace(1);
  m2.emplace_hint(m2.begin(), 1);
  for (const auto& k : m2) {
    (void)k;
  }
  (void)m2.find(Key{1});
  (void)m2.equal_range(Key{1});
  (void)(m2 == m2);
  m2.erase(m2.begin());
  m2.erase(m2.begin(), m2.end());
  std::erase_if(m2, []<class T>(const StrictComparable<T>&) -> BooleanTestable const& { return yes; });
}
