//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: stdlib=libc++

// <memory>
//
// unique_ptr
//
// constexpr pointer release() noexcept; // nodiscard as an extension

#include <memory>

void f(std::unique_ptr<int> p) {
  p.release(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
