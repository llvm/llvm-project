//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// template<class T>
// constexpr T& as-lvalue(T&& t) { // exposition only

#include <__utility/as_lvalue.h>

void test() {
  // Check prvalue
  {
    [[maybe_unused]] auto& check = std::__as_lvalue(
        0); // expected-warning {{temporary bound to local reference 'check' will be destroyed at the end of the full-expression}}
  }
}
