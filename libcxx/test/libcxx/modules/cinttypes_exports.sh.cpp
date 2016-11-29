//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// REQUIRES: modules-support

// Test that <cinttypes> re-exports <cstdint>

// RUN: %build_module

#include <cinttypes>

int main() {
  int8_t x; ((void)x);
  std::int8_t y; ((void)y);
}
