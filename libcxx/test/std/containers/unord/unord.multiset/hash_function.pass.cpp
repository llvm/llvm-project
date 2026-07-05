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
// class unordered_multiset

// hasher hash_function() const;

#include <unordered_set>
#include <cassert>

int main(int, char**) {
  typedef std::unordered_multiset<int> set_type;
  set_type s;

  set_type::iterator i1 = s.insert(1);
  set_type::iterator i2 = s.insert(1);

  const set_type& cs = s;
  assert(cs.hash_function()(*i1) == cs.hash_function()(*i2));
  assert(cs.hash_function()(*i2) == cs.hash_function()(*i1));

  return 0;
}
