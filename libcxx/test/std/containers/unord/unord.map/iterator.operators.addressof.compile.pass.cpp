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

// Validate the constructors of the (const)(_local)_iterator classes to be
// properly guarded against ADL-hijacking operator&.

#include <unordered_map>

#include "test_macros.h"
#include "operator_hijacker.h"

template <class ToIterator, class FromIterator>
void test() {
  FromIterator from;
  ToIterator copy(from);
  copy = from;

  ToIterator move(std::move(from));
  from = FromIterator();
  move = std::move(from);
}

void test() {
  {
    using I = std::unordered_map<operator_hijacker, operator_hijacker>::iterator;
    using CI = std::unordered_map<operator_hijacker, operator_hijacker>::const_iterator;
    test<I, I>();
    test<CI, I>();
    test<CI, CI>();
  }
  {
    using IL = std::unordered_map<operator_hijacker, operator_hijacker>::local_iterator;
    using CIL = std::unordered_map<operator_hijacker, operator_hijacker>::const_local_iterator;
    test<IL, IL>();
    test<CIL, IL>();
    test<CIL, CIL>();
  }
}
