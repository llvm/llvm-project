//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Test that `if constexpr` is provided as an extension by supported compilers
// in all language dialects after C++03. Also further test that the test suite
// has disabled -Wc++17-extension because we also disable system headers.

#include <cassert>

// if constexpr doesn't need to be used in a constexpr function, nor a dependent
// one.
bool CheckIfConstexpr() {
  if constexpr (false) {
    return false;
  }
  if constexpr (true) {
    return true;
  }
}

int main(int, char**) {
  assert(CheckIfConstexpr());
  return 0;
}
