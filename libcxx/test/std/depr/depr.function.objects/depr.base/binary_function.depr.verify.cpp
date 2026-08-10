//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// REQUIRES: c++11 || c++14
// binary_function was removed in C++17

// check that binary_function is marked deprecated

#include <functional>

std::binary_function<int, int, int> b; // expected-warning {{'binary_function<int, int, int>' is deprecated}}
