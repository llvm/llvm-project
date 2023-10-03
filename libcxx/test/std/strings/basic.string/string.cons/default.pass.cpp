//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string() noexcept(is_nothrow_default_constructible<allocator_type>::value); // constexpr since C++20

#include <cassert>
#include <string>

#include "test_macros.h"
#include "test_allocator.h"

#if TEST_STD_VER >= 11
// Test the noexcept specification, which is a conforming extension
LIBCPP_STATIC_ASSERT(std::is_nothrow_default_constructible<std::string>::value, "");
LIBCPP_STATIC_ASSERT(std::is_nothrow_default_constructible<
                         std::basic_string<char, std::char_traits<char>, test_allocator<char>>>::value,
                     "");
LIBCPP_STATIC_ASSERT(!std::is_nothrow_default_constructible<
                         std::basic_string<char, std::char_traits<char>, limited_allocator<char, 10>>>::value,
                     "");
#endif

TEST_CONSTEXPR_CXX20 bool test() {
  std::string str;
  assert(str.empty());

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
