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

// FIXME: The <atomic> header is not supported for single-threaded systems,
// but still gets built as part of the 'std' module, which breaks the build.
// XFAIL: libcpp-has-no-threads

// Test that <cstdint> re-exports <stdint.h>

// REQUIRES: modules-support
// ADDITIONAL_COMPILE_FLAGS: -fmodules

#include <cstdint>

int main(int, char**) {
  int8_t x; (void)x;
  std::int8_t y; (void)y;

  return 0;
}
