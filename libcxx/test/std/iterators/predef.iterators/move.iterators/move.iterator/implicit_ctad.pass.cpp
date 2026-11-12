//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <iterator>

// move_iterator

// Make sure that the implicitly-generated CTAD works.

#include <iterator>

#include "test_macros.h"

int main(int, char**) {
  int* it = nullptr;
  std::move_iterator move_it(it);
  ASSERT_SAME_TYPE(decltype(move_it), std::move_iterator<int*>);

  return 0;
}
