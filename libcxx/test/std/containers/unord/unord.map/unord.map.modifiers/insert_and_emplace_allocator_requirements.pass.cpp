//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// class unordered_map

// insert(...);
// emplace(...);

// UNSUPPORTED: c++98, c++03


#include <unordered_map>

#include "container_test_types.h"
#include "../../../map_allocator_requirement_test_templates.h"

int main()
{
  testMapInsert<TCT::unordered_map<> >();
  testMapInsertHint<TCT::unordered_map<> >();
  testMapEmplace<TCT::unordered_map<> >();
  testMapEmplaceHint<TCT::unordered_map<> >();
}
