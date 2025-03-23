//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <vector>

// template<class Allocator> struct hash<vector<bool, Allocator>>;

// libc++ makes the operator() of this partial specialization constexpr since C++20, which is a conforming extension.

#include <cassert>
#include <vector>

#include "min_allocator.h"
#include "poisoned_hash_helper.h"

constexpr bool test() {
  {
    using VB = std::vector<bool>;
    bool ba[]{true, false, true, true, false};
    VB vb(std::begin(ba), std::end(ba));

    const std::hash<VB> h{};
    const auto hash_value = h(vb);
    assert(hash_value == h(vb));
    assert(hash_value != 0);
  }
  {
    using VB  = std::vector<bool, min_allocator<bool>>;
    bool ba[] = {true, false, true, true, false};
    VB vb(std::begin(ba), std::end(ba));

    const std::hash<VB> h{};
    const auto hash_value = h(vb);
    assert(hash_value == h(vb));
    assert(hash_value != 0);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
