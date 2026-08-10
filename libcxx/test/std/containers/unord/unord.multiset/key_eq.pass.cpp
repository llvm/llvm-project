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

// key_equal key_eq() const;

#include <unordered_set>
#include <cassert>

int main(int, char**) {
  typedef std::unordered_multiset<int> set_type;
  set_type s;

  set_type::iterator i1 = s.insert(1);
  set_type::iterator i2 = s.insert(1);
  set_type::iterator i3 = s.insert(2);

  const set_type& cs = s;

  assert(cs.key_eq()(*i1, *i1));
  assert(cs.key_eq()(*i2, *i2));
  assert(cs.key_eq()(*i3, *i3));

  assert(cs.key_eq()(*i1, *i2));
  assert(cs.key_eq()(*i2, *i1));

  assert(!cs.key_eq()(*i1, *i3));
  assert(!cs.key_eq()(*i3, *i1));

  assert(!cs.key_eq()(*i2, *i3));
  assert(!cs.key_eq()(*i3, *i2));

  return 0;
}
