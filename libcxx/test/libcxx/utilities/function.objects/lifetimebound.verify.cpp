//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// ADDITIONAL_COMPILE_FLAGS: -Wno-pessimizing-move -Wno-unused-variable

#include <functional>

#include "test_macros.h"

// clang-format off

void func() {
  auto&& v1 = std::identity()(1); // expected-warning {{temporary bound to local reference 'v1' will be destroyed at the end of the full-expression}}
}
