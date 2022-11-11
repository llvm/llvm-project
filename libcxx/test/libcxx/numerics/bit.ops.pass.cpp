//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Test the __XXXX routines in the <bit> header.
// These are not supposed to be exhaustive tests, just sanity checks.

#include <bit>
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
