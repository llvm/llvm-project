//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// std::const_iterator

#include <iterator>
#include <list>
#include <ranges>
#include "test_macros.h"

ASSERT_SAME_TYPE(std::const_iterator<int*>, std::basic_const_iterator<int*>);
ASSERT_SAME_TYPE(std::const_iterator<const int*>, const int*);
ASSERT_SAME_TYPE(std::const_sentinel<int*>, std::basic_const_iterator<int*>);
ASSERT_SAME_TYPE(std::const_sentinel<const int*>, const int*);
ASSERT_SAME_TYPE(std::const_sentinel<std::default_sentinel_t>, std::default_sentinel_t);

using list_iterator       = std::list<int>::iterator;
using list_const_iterator = std::list<int>::const_iterator;

ASSERT_SAME_TYPE(std::const_iterator<list_iterator>, std::basic_const_iterator<list_iterator>);
ASSERT_SAME_TYPE(std::const_iterator<list_const_iterator>, list_const_iterator);
ASSERT_SAME_TYPE(std::const_sentinel<list_iterator>, std::basic_const_iterator<list_iterator>);
ASSERT_SAME_TYPE(std::const_sentinel<list_const_iterator>, list_const_iterator);

int main() { return 0; }
