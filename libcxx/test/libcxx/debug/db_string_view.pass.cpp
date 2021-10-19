//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: windows
// UNSUPPORTED: libcpp-no-if-constexpr

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1
// UNSUPPORTED: libcxx-no-debug-mode

// test container debugging

#include <string_view>

#include "test_macros.h"
#include "debug_mode_helper.h"

void test_null_argument() {
  // C++2b prohibits construction of string_view from nullptr_t.
  const char* nullp = nullptr;
  const char* null = NULL;
  (void)nullp;
  (void)null;
  EXPECT_DEATH((std::string_view(nullp)));
  EXPECT_DEATH((std::string_view(null)));
  EXPECT_DEATH(std::string_view(static_cast<const char*>(0)));
  {
    std::string_view v;
    EXPECT_DEATH(((void)(v == nullp)));
    EXPECT_DEATH(((void)(nullp == v)));
  }
}

int main(int, char**) {
  test_null_argument();

  return 0;
}
