//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: gcc

// XFAIL: has-no-cxx-module-support

// Apple Clang 17 advertises C++ Modules support but fails to compile this test.
// XFAIL: apple-clang-17

// A minimal test to validate import works.

// C++20 modules are incompatible with Clang modules
// ADDITIONAL_COMPILE_FLAGS: -fno-modules

// MODULE_DEPENDENCIES: std

import std;

int main(int, char**) {
  std::println("Hello modular world");
  return 0;
}
