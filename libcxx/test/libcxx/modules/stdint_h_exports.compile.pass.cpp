//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FIXME: The <atomic> header is not supported for single-threaded systems,
// but still gets built as part of the 'std' module, which breaks the build.
// The failure only shows up when modules are enabled AND we're building
// without threads, which is when the __config_site macro for _LIBCPP_HAS_NO_THREADS
// is honored.
// XFAIL: libcpp-has-no-threads && -fmodules

// Test that int8_t and the like are exported from stdint.h, not inttypes.h

// REQUIRES: modules-support
// ADDITIONAL_COMPILE_FLAGS: -fmodules

#include <stdint.h>

int main(int, char**) {
  int8_t x; (void)x;

  return 0;
}
