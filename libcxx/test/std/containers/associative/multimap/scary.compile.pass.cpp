//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map
// class multimap

// Extension: SCARY/N2913 iterator compatibility between map and multimap

#include <map>

#include "test_macros.h"

void test() {
  typedef std::map<int, int> M1;
  typedef std::multimap<int, int> M2;

  ASSERT_SAME_TYPE(M1::iterator, M2::iterator);
  ASSERT_SAME_TYPE(M1::const_iterator, M2::const_iterator);
}
