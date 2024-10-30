//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// <regex>

// template <class charT> struct regex_traits;

// locale_type getloc()const;

#include <regex>
#include <cassert>
#include <cstdio>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

int main(int, char**) {
  std::fprintf(stderr, "Entering main()\n");
  {
    std::fprintf(stderr, "Creating locale\n");
    std::locale loc("en_US.UTF-8");
    (void)loc;
  }
  return 0;
}
