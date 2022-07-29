//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that the default verbose termination function aborts the program.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// We flag uses of the verbose termination function in older dylibs at compile-time to avoid runtime
// failures when back-deploying.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}

#include <csignal>
#include <cstdlib>

void signal_handler(int signal) {
  if (signal == SIGABRT)
    std::_Exit(EXIT_SUCCESS);
  std::_Exit(EXIT_FAILURE);
}

int main(int, char**) {
  if (std::signal(SIGABRT, signal_handler) != SIG_ERR)
    _LIBCPP_ASSERT(false, "foo");
  return EXIT_FAILURE;
}
