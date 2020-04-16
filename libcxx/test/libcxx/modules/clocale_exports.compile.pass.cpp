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
// The failure only shows up when modules are enabled AND we're building
// without threads, which is when the __config_site macro for _LIBCPP_HAS_NO_THREADS
// is honored.
// XFAIL: libcpp-has-no-threads && -fmodules

// UNSUPPORTED: c++98, c++03

// REQUIRES: modules-support
// ADDITIONAL_COMPILE_FLAGS: -fmodules

#include <clocale>

int main(int, char**) {
  std::lconv l; (void)l;
  using T = decltype(std::setlocale(0, ""));
  using U = decltype(std::localeconv());

  return 0;
}
