//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__cxx03/__utility/no_destroy.h>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER > 17
// Test constexpr-constructibility.
constinit std::__no_destroy<int> nd_int_const(std::__uninitialized_tag{});
#endif

struct DestroyLast {
  ~DestroyLast() { assert(*ptr == 5); }

  int* ptr;
} last;

static std::__no_destroy<int> nd_int(5);

int main(int, char**) {
  last.ptr = &nd_int.__get();

  return 0;
}
