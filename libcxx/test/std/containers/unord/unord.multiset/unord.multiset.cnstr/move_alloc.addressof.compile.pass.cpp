//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_multiset

// Validate whether the operation properly guards against ADL-hijacking operator&

#include <unordered_set>

#include "test_allocator.h"
#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  using A = test_allocator<operator_hijacker>;
  using H = std::hash<operator_hijacker>;
  using P = std::equal_to<operator_hijacker>;

  const A a;
  std::unordered_multiset<operator_hijacker, H, P, A> so;
  std::unordered_multiset<operator_hijacker, H, P, A> s(std::move(so), a);
}
