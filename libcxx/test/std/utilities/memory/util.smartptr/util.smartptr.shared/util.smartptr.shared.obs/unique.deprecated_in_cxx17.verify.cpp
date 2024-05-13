//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_SHARED_PTR_UNIQUE

// <memory>

// shared_ptr

// bool unique() const; // deprecated in C++17, removed in C++20

#include <memory>

void f() {
  const std::shared_ptr<int> p;
  p.unique(); // expected-warning {{'unique' is deprecated}}
}
