//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-exceptions

// <inplace_vector>

// constexpr void resize(size_type sz);

#include <cassert>
#include <inplace_vector>

#include "../common.h"

int main(int, char**) {
  ThrowingValue::reset();
  {
    std::inplace_vector<ThrowingValue, 4> c;
    c.emplace_back(1);
    c.emplace_back(2);

    ThrowingValue::throw_after = 1;
    try {
      c.resize(4);
      assert(false);
    } catch (int) {
      assert(c.size() == 2);
      assert(c[0].value == 1);
      assert(c[1].value == 2);
      assert(ThrowingValue::alive == 2);
    }
  }
  assert(ThrowingValue::alive == 0);

  return 0;
}
