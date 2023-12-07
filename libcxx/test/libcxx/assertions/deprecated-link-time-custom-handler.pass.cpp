//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: availability-verbose_abort-missing

// Make sure that we still support _LIBCPP_AVAILABILITY_CUSTOM_VERBOSE_ABORT_PROVIDED for folks
// who customize the verbose termination function at link-time in back-deployment environments.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_AVAILABILITY_CUSTOM_VERBOSE_ABORT_PROVIDED

// We emit a #warning about the deprecation of this setting, so make sure we don't turn that into an error.
// ADDITIONAL_COMPILE_FLAGS: -Wno-error

#include <cstdlib>

void std::__libcpp_verbose_abort(char const*, ...) {
  std::exit(EXIT_SUCCESS);
}

int main(int, char**) {
  _LIBCPP_ASSERT(false, "message");
  return EXIT_FAILURE;
}
