//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// Validate various member functions of std::inplace_vector with an ADL-hijacking operator&.

#include <inplace_vector>
#include <utility>

#include "operator_hijacker.h"
#include "test_iterators.h"

using InplaceVector = std::inplace_vector<operator_hijacker, 16>;

void test(
    InplaceVector v,
    InplaceVector::const_iterator it,
    cpp17_input_iterator<operator_hijacker*> other_it,
    operator_hijacker val) {
  // emplace / insert
  v.emplace(it);
  v.insert(it, it, it);
  v.insert(it, other_it, other_it);
  v.insert(it, operator_hijacker());
  v.insert(it, 1, val);
  v.insert(it, val);

  // erase
  v.erase(it);
  v.erase(it, it);

  // assignment
  v = static_cast<InplaceVector&>(v);
  v = std::move(v);

  // construction
  {
    InplaceVector v2(std::move(v));
    (void)v2;
  }

  // swap
  v.swap(v);
}
