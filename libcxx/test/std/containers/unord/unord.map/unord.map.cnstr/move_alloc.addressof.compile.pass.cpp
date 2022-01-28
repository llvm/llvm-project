//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// unordered_map(unordered_map&& u, const allocator_type& a);

// Validate whether the operation properly guards against ADL-hijacking operator&

#include <unordered_map>

#include "test_macros.h"
#include "operator_hijacker.h"

#include "test_allocator.h"
#include "min_allocator.h"

void test() {
  using A = test_allocator<std::pair<const operator_hijacker, operator_hijacker>>;
  using C = std::unordered_map<operator_hijacker, operator_hijacker, std::hash<operator_hijacker>,
                               std::equal_to<operator_hijacker>, A>;

  C mo;
  C m(std::move(mo), A());
}
