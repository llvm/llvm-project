//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-exceptions

// <vector>

// template <class... Args>
//     reference emplace_back(Args&&... args); // reference in C++17

#include <cassert>
#include <vector>

#include "test_allocator.h"

int main(int, char**) {
  std::vector<int, limited_allocator<int, 10> > v(10, 42);
  try {
    v.emplace_back(0);
    assert(false);
  } catch (...) {
    assert(v.size() == v.max_size());
    for (std::size_t i = 0; i != v.size(); ++i)
      assert(v[i] == 42); // Strong exception safety guarantee
  }

  return 0;
}
