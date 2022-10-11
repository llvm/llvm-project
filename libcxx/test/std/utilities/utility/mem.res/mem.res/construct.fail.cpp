//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <memory_resource>

// Check that memory_resource is not constructible

#include <memory_resource>

void test() {
  std::pmr::memory_resource m; // expected-error {{variable type 'std::pmr::memory_resource' is an abstract class}}
}
