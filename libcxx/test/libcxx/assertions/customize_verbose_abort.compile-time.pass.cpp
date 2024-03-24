//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This compile-time customization requires cross-file macros, which doesn't work with modules.
// UNSUPPORTED: clang-modules-build

// Make sure that we can customize the verbose termination function at compile-time by
// defining _LIBCPP_VERBOSE_ABORT ourselves. Note that this does not have any
// deployment target requirements.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_VERBOSE_ABORT(...)=my_abort(__VA_ARGS__)

#include <cstdlib>

void my_abort(char const*, ...) {
  std::exit(EXIT_SUCCESS);
}

int main(int, char**) {
  _LIBCPP_VERBOSE_ABORT("%s", "message");
  return EXIT_FAILURE;
}
