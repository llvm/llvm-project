//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=0
// UNSUPPORTED: libcxx-no-debug-mode

// Test that the default debug handler aborts the program.

#include <csignal>
#include <cstdlib>
#include <__debug>

#include "test_macros.h"

void signal_handler(int signal)
{
    if (signal == SIGABRT)
      std::_Exit(EXIT_SUCCESS);
    std::_Exit(EXIT_FAILURE);
}

int main(int, char**)
{
  if (std::signal(SIGABRT, signal_handler) != SIG_ERR)
    _LIBCPP_ASSERT(false, "foo");
  return EXIT_FAILURE;
}
