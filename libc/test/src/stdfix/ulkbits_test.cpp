//===-- Unittests for ulkbits ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FxBitsTest.h"
#include "src/stdfix/ulkbits.h"

LIST_FXBITS_TEST(ulk, unsigned long accum, uint_ulk_t, LIBC_NAMESPACE::ulkbits);
