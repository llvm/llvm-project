//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: modules-support

// Test that int8_t and the like are exported from stdint.h not inttypes.h

// RUN: %build_module

#include <stdint.h>

int main(int, char**) {
  int8_t x; ((void)x);

  return 0;
}
