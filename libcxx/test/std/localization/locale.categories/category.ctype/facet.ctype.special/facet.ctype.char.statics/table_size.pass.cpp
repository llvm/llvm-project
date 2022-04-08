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

// Before https://llvm.org/D110647, the shared library did not contain
// std::ctype<char>::table_size, so this test fails with a link error.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{11.0|12.0|13.0}}

#include <locale>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  typedef std::ctype<char> F;
  const size_t* G = &F::table_size;
  assert(*G >= 256);

  return 0;
}
