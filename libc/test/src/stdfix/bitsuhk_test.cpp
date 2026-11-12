//===-- Unittests for bitsuhk ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/src/stdfix/BitsFxTest.h"

#include "include/llvm-libc-types/uint_uhk_t.h" // uint_uhk_t
#include "src/stdfix/bitsuhk.h"

LIST_BITSFX_TESTS(uhk, unsigned short accum, uint_uhk_t,
                  LIBC_NAMESPACE::bitsuhk);
