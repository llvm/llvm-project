//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// hasher key_eq() const;

#include <unordered_map>
#include <string>
#include <cassert>

int main(int, char**) {
  typedef std::unordered_map<int, std::string> map_type;

  map_type m;
  std::pair<map_type::iterator, bool> p1 = m.insert(map_type::value_type(1, "abc"));
  std::pair<map_type::iterator, bool> p2 = m.insert(map_type::value_type(2, "abc"));

  const map_type& cm = m;

  assert(cm.key_eq()(p1.first->first, p1.first->first));
  assert(cm.key_eq()(p2.first->first, p2.first->first));
  assert(!cm.key_eq()(p1.first->first, p2.first->first));
  assert(!cm.key_eq()(p2.first->first, p1.first->first));

  return 0;
}
