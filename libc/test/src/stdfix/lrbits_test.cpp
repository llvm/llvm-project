//===-- Unittests for lrbits ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FxBitsTest.h"
#include "src/stdfix/lrbits.h"

LIST_FXBITS_TEST(lr, long fract, int_lr_t, LIBC_NAMESPACE::lrbits);
