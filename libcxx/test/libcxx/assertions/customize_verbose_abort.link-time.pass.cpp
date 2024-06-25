//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that we can set a custom verbose termination function at link-time.

// We flag uses of the verbose termination function in older dylibs at compile-time to avoid runtime
// failures when back-deploying.
// XFAIL: availability-verbose_abort-missing

#include <__verbose_abort>
#include <cstdlib>

void std::__libcpp_verbose_abort(char const*, ...) {
  std::exit(EXIT_SUCCESS);
}

int main(int, char**) {
  std::__libcpp_verbose_abort("%s", "message");
  return EXIT_FAILURE;
}
