//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class ctype<char>

// Make sure we can reference std::ctype<char>::table_size.

// Before https://llvm.org/D110647 (049f6c29a6f0), the shared library did not contain
// std::ctype<char>::table_size, so this test fails with a link error.
// XFAIL: using-built-library-before-llvm-14

#include <locale>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  typedef std::ctype<char> F;
  const std::size_t* G = &F::table_size;
  assert(*G >= 256);

  return 0;
}
