//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// template <class... Args>
//     iterator try_emplace(const_iterator hint, const key_type& k, Args&&... args);
// template <class... Args>
//     iterator try_emplace(const_iterator hint, key_type&& k, Args&&... args);
// template <class M>

// Validate whether the operation properly guards against ADL-hijacking operator&

#include <unordered_map>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  std::unordered_map<operator_hijacker, operator_hijacker> m;
  {
    const operator_hijacker k;
    m.try_emplace(m.cend(), k);
  }
  {
    operator_hijacker k;
    m.try_emplace(m.cend(), std::move(k));
  }
}
