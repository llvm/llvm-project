//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// This test ensures that std::basic_string handles allocator types with small size types
// properly.

#include <cassert>
#include <cstdint>
#include <string>

#include "sized_allocator.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool tests() {
  {
    using Alloc = sized_allocator<char, std::uint8_t, std::int8_t>;
    std::basic_string<char, std::char_traits<char>, Alloc> s(Alloc(1));
    s.append(30, 'a'); // growth: __get_amortized_growth_capacity
    assert(s.size() == 30);
    assert(s.capacity() >= 30);
  }
  {
    using Alloc = sized_allocator<char, std::uint8_t, std::int8_t>;
    std::basic_string<char, std::char_traits<char>, Alloc> s("abcdef", Alloc(1));
    s.replace(2, 100, "XY", 2); // n1 clamped to size() - pos
    assert(s.size() == 4);
  }
  {
    using Alloc = sized_allocator<char, std::uint16_t, std::int16_t>;
    std::basic_string<char, std::char_traits<char>, Alloc> s("abcdef", Alloc(1));
    s.replace(2, 100, 3, 'Z'); // n1 clamped to size() - pos
    assert(s.size() == 5);
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
