//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_set

// hasher hash_function() const;

#include <unordered_set>
#include <cassert>

int main(int, char**) {
  typedef std::unordered_set<int> set_type;
  set_type s;

  std::pair<set_type::iterator, bool> p = s.insert(1);

  const set_type& cs = s;
  assert(cs.hash_function()(*p.first) == cs.hash_function()(1));
  assert(cs.hash_function()(1) == cs.hash_function()(*p.first));

  return 0;
}
