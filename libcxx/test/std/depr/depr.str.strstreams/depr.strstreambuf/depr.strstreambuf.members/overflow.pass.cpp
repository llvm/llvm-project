//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstreambuf

// int overflow(int c);

// There was an overflow in the dylib on older macOS versions
// UNSUPPORTED: with_system_cxx_lib=macosx10.8
// UNSUPPORTED: with_system_cxx_lib=macosx10.7

#include <iostream>
#include <string>
#include <strstream>

#include "test_macros.h"

int main(int, char**) {
  std::ostrstream oss;
  std::string s;

  for (int i = 0; i < 4096; ++i)
    s.push_back((i % 16) + 'a');

  oss << s << std::ends;
  std::cout << oss.str();
  oss.freeze(false);

  return 0;
}
