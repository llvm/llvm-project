//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: no-exceptions

// <vector>
// vector<bool>

// template <class... Args>
//     reference emplace_back(Args&&... args); // reference in C++17

#include <cassert>
#include <stdexcept>
#include <vector>

#include "test_allocator.h"

int main(int, char**) {
  std::vector<bool, limited_allocator<bool, 10> > v;
  v.resize(v.max_size(), true);

  // Attempt to append one more element to a vector<bool> that is already at its maximum possible size
  try {
    v.emplace_back(true);
    assert(false);
  } catch (const std::length_error&) {
    assert(v.size() == v.max_size());
    for (std::size_t i = 0; i != v.size(); ++i)
      assert(v[i] == true);
  }

  return 0;
}
