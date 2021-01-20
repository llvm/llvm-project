//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Some headers are not available when these features are disabled, but they
// still get built as part of the 'std' module, which breaks the build.
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: libcpp-has-no-localization
// UNSUPPORTED: libcpp-has-no-filesystem-library

// Test that int8_t and the like are exported from stdint.h, not inttypes.h

// REQUIRES: modules-support
// ADDITIONAL_COMPILE_FLAGS: -fmodules

#include <stdint.h>

int main(int, char**) {
  int8_t x; (void)x;

  return 0;
}
