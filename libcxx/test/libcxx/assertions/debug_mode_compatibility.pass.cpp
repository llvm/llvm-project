//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that assertions are still enabled when _LIBCPP_DEBUG=0 is
// defined, for backwards compatibility with code that might have been using
// it to enable assertions previously.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=0

// We flag uses of the assertion handler in older dylibs at compile-time to avoid runtime
// failures when back-deploying.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}

#include <cstdlib>

void std::__libcpp_verbose_abort(char const*, ...) {
  std::exit(EXIT_SUCCESS);
}

int main(int, char**) {
  _LIBCPP_ASSERT(false, "message");
  return EXIT_FAILURE;
}
