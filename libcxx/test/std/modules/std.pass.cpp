//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// TODO MODULES fix c++26
// XFAIL: c++26

// REQUIRES: use_module_std

// A minimal test to validate import works.

import std;

int main(int, char**) {
  std::println("Hello modular world");
  return 0;
}
