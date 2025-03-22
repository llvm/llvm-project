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

// void reserve(size_type n);

#include <cassert>
#include <stdexcept>
#include <vector>

#include "test_allocator.h"

void test() {
  {
    std::vector<bool, limited_allocator<bool, 10> > v;
    v.reserve(5);
    try {
      // A typical implementation would allocate chunks of bits.
      // In libc++ the chunk has the same size as the machine word. It is
      // reasonable to assume that in practice no implementation would use
      // 64 kB or larger chunks.
      v.reserve(10 * 65536);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.empty());
      assert(v.capacity() >= 5);
    }
  }
  {
    using Vec = std::vector<bool, limited_allocator<bool, 10> >;
    Vec v(Vec().max_size(), true);
    try {
      v.reserve(v.max_size() + 1);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i < v.size(); ++i)
        assert(v[i] == true);
    }
  }
  {
    bool a[] = {true, false, true, false, true};
    std::vector<bool> v(std::begin(a), std::end(a));
    try {
      v.reserve(v.max_size() + 1);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == 5);
      assert(v.capacity() >= 5);
      for (std::size_t i = 0; i < v.size(); ++i)
        assert(v[i] == a[i]);
    }
  }
}

int main(int, char**) {
  test();
  return 0;
}
