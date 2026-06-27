//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

// insert(...)

// UNSUPPORTED: c++03

#include <set>

#include "test_macros.h"
#include "container_test_types.h"
#include "../../set_allocator_requirement_test_templates.h"

TEST_CONSTEXPR_CXX26 bool test() {
  testMultisetInsert<TCT::multiset<> >();
  testMultisetEmplace<TCT::multiset<> >();

  return true;
}
int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
