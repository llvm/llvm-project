//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// This test ensures that std::vector<bool> handles allocator types with small size types
// properly. Related issue: https://llvm.org/PR121713.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "sized_allocator.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool tests() {
  {
    using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
    std::vector<bool, Alloc> c(Alloc(1));
    c.resize(10);
    assert(c.size() == 10);
    assert(c.capacity() >= 10);
  }
  {
    using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
    std::vector<bool, Alloc> c(Alloc(1));
    c.assign(10, true);
    assert(c.size() == 10);
    assert(c.capacity() >= 10);
  }
  {
    using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
    std::vector<bool, Alloc> c(Alloc(1));
    c.insert(c.end(), true);
    assert(c.size() == 1);
    assert(c.capacity() >= 1);
  }
  {
    using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
    std::vector<bool, Alloc> c(Alloc(1));
    c.insert(c.end(), 10, true);
    assert(c.size() == 10);
    assert(c.capacity() >= 10);
  }
  {
    using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
    std::vector<bool, Alloc> c(Alloc(1));
    c.push_back(true);
    assert(c.size() == 1);
    assert(c.capacity() >= 1);
  }
  {
    using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
    std::vector<bool, Alloc> c(Alloc(1));
    c.resize(10, true);
    assert(c.size() == 10);
    assert(c.capacity() >= 10);
  }
  {
    using Alloc = sized_allocator<bool, std::uint32_t, std::int32_t>;
    std::vector<bool, Alloc> c(Alloc(1));
    c.resize(10);
    assert(c.size() == 10);
    assert(c.capacity() >= 10);
  }
  {
    using Alloc = sized_allocator<bool, std::uint64_t, std::int64_t>;
    std::vector<bool, Alloc> c(Alloc(1));
    c.resize(10);
    assert(c.size() == 10);
    assert(c.capacity() >= 10);
  }
  {
    using Alloc = sized_allocator<bool, std::size_t, std::ptrdiff_t>;
    std::vector<bool, Alloc> c(Alloc(1));
    c.resize(10);
    assert(c.size() == 10);
    assert(c.capacity() >= 10);
  }

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER >= 20
  static_assert(tests());
#endif
  return 0;
}
