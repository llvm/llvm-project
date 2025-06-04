//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// class unordered_set
// class unordered_multiset

// Extension: SCARY/N2913 iterator compatibility between unordered_set and unordered_multiset

#include <unordered_set>

#include "test_macros.h"

void test() {
  typedef std::unordered_set<int> M1;
  typedef std::unordered_multiset<int> M2;

  ASSERT_SAME_TYPE(M1::iterator, M2::iterator);
  ASSERT_SAME_TYPE(M1::const_iterator, M2::const_iterator);
  ASSERT_SAME_TYPE(M1::local_iterator, M2::local_iterator);
  ASSERT_SAME_TYPE(M1::const_local_iterator, M2::const_local_iterator);
}
