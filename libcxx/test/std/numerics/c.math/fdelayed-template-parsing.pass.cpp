//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that cmath builds with -fdelayed-template-parsing.
// This is a regression test for an issue introduced in ae22f0b24231,
// where Clang's limited support for -fdelayed-template-parsing would
// choke on <cmath>.

// REQUIRES: fdelayed-template-parsing
// ADDITIONAL_COMPILE_FLAGS: -fdelayed-template-parsing

#include <cmath>
#include <cassert>

int main(int, char**) {
  assert(std::isfinite(1.0));
  assert(!std::isinf(1.0));
  assert(!std::isnan(1.0));

  return 0;
}

using namespace std; // on purpose
