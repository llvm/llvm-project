//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -O0 -g
// UNSUPPORTED: no-localization
// XFAIL: availability-stacktrace-missing

/*
This isn't really a test; it simply takes a stacktrace and prints to stdout,
so we can see what a stacktrace actually contains and looks like when printed.
*/

#include <__config_site>
#if _LIBCPP_HAS_LOCALIZATION

#  include <cassert>
#  include <iostream>
#  include <stacktrace>

int main(int, char**) {
  std::cout << std::stacktrace::current() << '\n';
  return 0;
}

#else
int main() { return 0; }
#endif // _LIBCPP_HAS_LOCALIZATION
