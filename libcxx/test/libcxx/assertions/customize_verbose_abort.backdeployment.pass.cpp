//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that we can enable assertions when we back-deploy to older platforms
// if we define _LIBCPP_AVAILABILITY_CUSTOM_VERBOSE_ABORT_PROVIDED.
//
// Note that this test isn't really different from customize_verbose_abort.pass.cpp when
// run outside of back-deployment scenarios, but we always want to run this test.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1 -D_LIBCPP_AVAILABILITY_CUSTOM_VERBOSE_ABORT_PROVIDED

#include <cstdlib>

void std::__libcpp_verbose_abort(char const*, ...) {
  std::exit(EXIT_SUCCESS);
}

int main(int, char**) {
  _LIBCPP_ASSERT(false, "message");
  return EXIT_FAILURE;
}
