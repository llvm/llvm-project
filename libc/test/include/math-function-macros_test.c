//===-- Unittests for math function macros --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/math-function-macros.h"

#include <assert.h>

#define CHECK_FUNCTION(FUNC_NAME, VAL)                                         \
  do {                                                                         \
    assert(FUNC_NAME((unsigned char)0U) == VAL##AU);                           \
    assert(FUNC_NAME((unsigned short)0U) == VAL##BU);                          \
    assert(FUNC_NAME(0U) == VAL##CU);                                          \
    assert(FUNC_NAME(0UL) == VAL##DU);                                         \
    assert(FUNC_NAME(0ULL) == VAL##EU);                                        \
  } while (0)

int main(void) {
  CHECK_FUNCTION(stdc_count_zeros, 0x2);
  //CHECK_FUNCTION(isfinite(pos_inf), 0);

  /*
  assert(!stdc_has_single_bit((unsigned char)1U));
  assert(!stdc_has_single_bit((unsigned short)1U));
  assert(!stdc_has_single_bit(1U));
  assert(!stdc_has_single_bit(1UL));
  assert(!stdc_has_single_bit(1ULL));
  */

  return 0;
}
