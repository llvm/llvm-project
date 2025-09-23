//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test the __XXXX routines in the <bit> header.
// These are not supposed to be exhaustive tests, just sanity checks.

#include <__bit/bit_log2.h>
#include <__bit/countl.h>
#include <__bit/rotate.h>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX14 bool test() {
  const unsigned v = 0x12345678;

  ASSERT_SAME_TYPE(unsigned, decltype(std::__rotr(v, 3)));
  ASSERT_SAME_TYPE(int, decltype(std::__countl_zero(v)));

  assert(std::__rotr(v, 3) == 0x02468acfU);
  assert(std::__countl_zero(v) == 3);

#if TEST_STD_VER > 17
  ASSERT_SAME_TYPE(unsigned, decltype(std::__bit_log2(v)));
  assert(std::__bit_log2(v) == 28);
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 11
  static_assert(test(), "");
#endif

  return 0;
}
