//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// REQUIRES: modules-support

// Test that intypes.h re-exports stdint.h

// RUN: %build_module

#include <inttypes.h>

int main() {
  int8_t x; ((void)x);
}
