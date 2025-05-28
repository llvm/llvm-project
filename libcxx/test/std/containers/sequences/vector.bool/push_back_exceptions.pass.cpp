//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// <vector>
// vector<bool>

// void push_back(const value_type& x);

#include <cassert>
#include <vector>

#include "test_allocator.h"

int main(int, char**) {
  using Vec = std::vector<bool, limited_allocator<bool, 10> >;
  Vec v(Vec().max_size(), true);
  try {
    v.push_back(true);
    assert(false);
  } catch (...) {
    assert(v.size() == v.max_size());
    for (std::size_t i = 0; i != v.size(); ++i)
      assert(v[i] == true);
  }

  return 0;
}
