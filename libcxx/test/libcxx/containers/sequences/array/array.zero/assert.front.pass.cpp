//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, windows
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// test that array<T, 0>::back() triggers an assertion

#include <array>

#include "check_assertion.h"

int main(int, char**) {
  {
    typedef std::array<int, 0> C;
    C c = {};
    C const& cc = c;
    TEST_LIBCPP_ASSERT_FAILURE(c.front(), "cannot call array<T, 0>::front() on a zero-sized array");
    TEST_LIBCPP_ASSERT_FAILURE(cc.front(), "cannot call array<T, 0>::front() on a zero-sized array");
  }
  {
    typedef std::array<const int, 0> C;
    C c = {{}};
    C const& cc = c;
    TEST_LIBCPP_ASSERT_FAILURE(c.front(), "cannot call array<T, 0>::front() on a zero-sized array");
    TEST_LIBCPP_ASSERT_FAILURE(cc.front(), "cannot call array<T, 0>::front() on a zero-sized array");
  }

  return 0;
}
