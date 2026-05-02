//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <flat_map>
//
// flat_map should support comparator that return a non-boolean
// value as long as the returned type is implicitly convertible to bool.

#include <flat_map>
#include <vector>
#include <ranges>

#include "boolean_testable.h"

void test() {
  using Key   = StrictComparable<int>;
  using Value = StrictComparable<int>;
  std::flat_map<Key, Value> m1;
  std::flat_map m2(std::from_range, m1, StrictBinaryPredicate);
  std::flat_map m3(std::sorted_unique, m1.keys(), m1.values(), StrictBinaryPredicate);
  std::flat_map m4(m1.begin(), m1.end(), StrictBinaryPredicate);
  m2.insert(m1.begin(), m1.end());
  m2.insert(std::sorted_unique, m1.begin(), m1.end());
  m2.insert_range(m1);
  (void)m2.at(2);
  m3[1] = 2;
  m3.insert_or_assign(1, 2);
  m4.try_emplace(1, 2);
  m2.emplace(1, 2);
  m2.emplace_hint(m2.begin(), 1, 2);
  for (const auto& [k, v] : m2) {
    (void)k;
    (void)v;
  }
  (void)m2.find(Key{1});
  (void)m2.equal_range(Key{1});
  (void)(m2 == m2);
  m2.erase(m2.begin());
  m2.erase(m2.begin(), m2.end());
  std::erase_if(
      m2, []<class T>(std::pair<const StrictComparable<T>&, const StrictComparable<T>&>) -> BooleanTestable const& {
        return yes;
      });
}
