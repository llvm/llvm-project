//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// basic_string(basic_string&& str) noexcept; // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

int main(int, char**) {
  {
    typedef std::string C;
    static_assert(std::is_nothrow_move_constructible<C>::value, "");
  }
  {
    typedef std::basic_string<char, std::char_traits<char>, test_allocator<char>> C;
    static_assert(std::is_nothrow_move_constructible<C>::value, "");
  }
  {
    typedef std::basic_string<char, std::char_traits<char>, limited_allocator<char, 10>> C;
    static_assert(std::is_nothrow_move_constructible<C>::value, "");
  }
  {
    using C = std::basic_string<char, std::char_traits<char>, noexcept_false_ctor_allocator<char>>;
    static_assert(std::is_nothrow_move_constructible<C>::value, "");
  }

  return 0;
}
