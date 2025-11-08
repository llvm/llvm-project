//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <fstream>

// Check that functions are marked [[nodiscard]]

#include <fstream>

void test() {
  {
    std::filebuf fb;
    fb.native_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::fstream fs;
    fs.native_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::ifstream fs;
    fs.native_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::ofstream fs;
    fs.native_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
}
