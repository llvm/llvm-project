//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// pointer_safety get_pointer_safety();

// UNSUPPORTED: c++98, c++03

#include <memory>
#include <cassert>

#include "test_macros.h"


void test_pr26961() {
  std::pointer_safety d;
  d = std::get_pointer_safety();
  assert(d == std::get_pointer_safety());
}

int main(int, char**)
{
  {
    std::pointer_safety r = std::get_pointer_safety();
    assert(r == std::pointer_safety::relaxed ||
           r == std::pointer_safety::preferred ||
           r == std::pointer_safety::strict);
  }
  {
    test_pr26961();
  }

  return 0;
}
