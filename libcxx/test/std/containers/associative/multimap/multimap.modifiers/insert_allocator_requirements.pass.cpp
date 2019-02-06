//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// insert(...)

// UNSUPPORTED: c++98, c++03

#include <map>

#include "container_test_types.h"
#include "../../../map_allocator_requirement_test_templates.h"


int main()
{
  testMultimapInsert<TCT::multimap<> >();
  testMultimapInsertHint<TCT::multimap<> >();
}
