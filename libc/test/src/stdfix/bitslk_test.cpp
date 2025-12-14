//===-- Unittests for bitslk ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitsFxTest.h"

#include "llvm-libc-types/int_lk_t.h" // int_lk_t
#include "src/stdfix/bitslk.h"

LIST_BITSFX_TESTS(lk, long accum, int_lk_t, LIBC_NAMESPACE::bitslk);
