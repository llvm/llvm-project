//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test fails on Windows because the underlying libc headers on Windows
// are not modular
// XFAIL: LIBCXX-WINDOWS-FIXME

// Some headers are not available when these features are disabled, but they
// still get built as part of the 'std' module, which breaks the build.
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: libcpp-has-no-localization

// REQUIRES: modules-support
// ADDITIONAL_COMPILE_FLAGS: -fmodules

// Test that <cinttypes> re-exports <cstdint>

#include <cinttypes>

int main(int, char**) {
  int8_t x; (void)x;
  std::int8_t y; (void)y;

  return 0;
}
