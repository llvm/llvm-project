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

// hasher hash_function() const;

#include <unordered_map>
#include <string>
#include <cassert>

int main(int, char**) {
  typedef std::unordered_map<int, std::string> map_type;
  map_type m;

  std::pair<map_type::iterator, bool> p = m.insert(map_type::value_type(1, "abc"));

  const map_type& cm = m;
  assert(cm.hash_function()(p.first->first) == cm.hash_function()(1));
  assert(cm.hash_function()(1) == cm.hash_function()(p.first->first));

  return 0;
}
