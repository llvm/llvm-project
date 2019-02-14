//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// class unordered_multiset

// insert(...)

// UNSUPPORTED: c++98, c++03

#include <unordered_set>
#include "container_test_types.h"
#include "../../set_allocator_requirement_test_templates.h"

int main(int, char**)
{
  testMultisetInsert<TCT::unordered_multiset<> >();
  testMultisetEmplace<TCT::unordered_multiset<> >();

  return 0;
}
