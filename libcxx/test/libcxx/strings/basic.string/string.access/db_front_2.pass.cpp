//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Call front() on empty container.

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <string>
#include <cassert>

#include "test_macros.h"
#include "debug_macros.h"
#include "min_allocator.h"

int main(int, char**) {
  typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
  S s(1, '\0');
  assert(s.front() == 0);
  s.clear();
  TEST_LIBCPP_ASSERT_FAILURE(s.front(), "string::front(): string is empty");

  return 0;
}
