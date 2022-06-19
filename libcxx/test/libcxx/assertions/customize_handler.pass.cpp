//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that we can set a custom assertion handler.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// We flag uses of the assertion handler in older dylibs at compile-time to avoid runtime
// failures when back-deploying.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}

#include <cassert>

bool handler_called = false;
void std::__libcpp_assertion_handler(char const*, int, char const*, char const*) {
  handler_called = true;
}

int main(int, char**) {
  _LIBCPP_ASSERT(false, "message");
  assert(handler_called);
  return 0;
}
