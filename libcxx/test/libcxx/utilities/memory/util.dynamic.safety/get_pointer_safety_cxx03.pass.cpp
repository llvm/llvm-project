//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// pointer_safety get_pointer_safety();

#include <memory>
#include <cassert>

#include "test_macros.h"

// libc++ doesn't offer std::pointer_safety in C++03 under the new ABI
#if TEST_STD_VER < 11 && defined(_LIBCPP_ABI_POINTER_SAFETY_ENUM_TYPE)
#define TEST_IS_UNSUPPORTED
#endif

#ifndef TEST_IS_UNSUPPORTED
void test_pr26961() {
  std::pointer_safety d;
  d = std::get_pointer_safety();
  assert(d == std::get_pointer_safety());
}
#endif

int main()
{
#ifndef TEST_IS_UNSUPPORTED
  {
    // Test that std::pointer_safety is still offered in C++03 under the old ABI.
    std::pointer_safety r = std::get_pointer_safety();
    assert(r == std::pointer_safety::relaxed ||
           r == std::pointer_safety::preferred ||
           r == std::pointer_safety::strict);
  }
  {
    test_pr26961();
  }
#endif
}
