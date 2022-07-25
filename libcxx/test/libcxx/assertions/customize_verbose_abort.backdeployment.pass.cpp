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
// Note that this test isn't really different from customize_handler.pass.cpp when
// run outside of back-deployment scenarios, but we still run it all the time.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1 -D_LIBCPP_AVAILABILITY_CUSTOM_VERBOSE_ABORT_PROVIDED

#include <cassert>

bool handler_called = false;
void std::__libcpp_verbose_abort(char const*, ...) {
  handler_called = true;
}

int main(int, char**) {
  _LIBCPP_ASSERT(false, "message");
  assert(handler_called);
  return 0;
}
