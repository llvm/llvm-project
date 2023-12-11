//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// UNSUPPORTED: libcpp-has-no-std-modules
// UNSUPPORTED: clang-modules-build

// A minimal test to validate import works.

import std.compat;

int main(int, char**) { return !(::strlen("Hello modular world") == 19); }
