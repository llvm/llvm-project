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

// REQUIRES: modules-support
// UNSUPPORTED: c++98, c++03

// RUN: %build_module

#include <clocale>

#define TEST(...) do { using T = decltype( __VA_ARGS__ ); } while(false)

int main(int, char**) {
  std::lconv l; ((void)l);

  TEST(std::setlocale(0, ""));
  TEST(std::localeconv());

  return 0;
}
