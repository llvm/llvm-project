//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// class unordered_multimap

// insert(...)

// UNSUPPORTED: c++98, c++03

#include <unordered_map>

#include "container_test_types.h"
#include "../../../map_allocator_requirement_test_templates.h"

int main(int, char**)
{
  testMultimapInsert<TCT::unordered_multimap<> >();
  testMultimapInsertHint<TCT::unordered_multimap<> >();

  return 0;
}
