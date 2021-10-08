//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Guard the debug iterators against ADL-hijacking.
// XFAIL: LIBCXX-DEBUG-FIXME

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_multimap

// unordered_multimap& operator=(const unordered_multimap& u);

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <unordered_map>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  {
    std::unordered_multimap<int, operator_hijacker> mo;
    std::unordered_multimap<int, operator_hijacker> m;
    m = mo;
  }
  {
    std::unordered_multimap<operator_hijacker, int> mo;
    std::unordered_multimap<operator_hijacker, int> m;
    m = mo;
  }
}
