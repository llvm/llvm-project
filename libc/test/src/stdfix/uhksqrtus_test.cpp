//===-- Unittests for uhksqrtus -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ISqrtTest.h"

#include "src/__support/fixed_point/sqrt.h"
#include "src/stdfix/uhksqrtus.h"

unsigned short accum uhksqrtus_fast(unsigned short x) {
  return LIBC_NAMESPACE::fixed_point::isqrt_fast(x);
}

LIST_ISQRT_TESTS(US, unsigned short, LIBC_NAMESPACE::uhksqrtus);

LIST_ISQRT_TESTS(USFast, unsigned short, uhksqrtus_fast);
