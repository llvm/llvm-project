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

// UNSUPPORTED: c++03

// REQUIRES: modules-support
// ADDITIONAL_COMPILE_FLAGS: -fmodules

#include <clocale>

int main(int, char**) {
  std::lconv l; (void)l;
  using T = decltype(std::setlocale(0, ""));
  using U = decltype(std::localeconv());

  return 0;
}
