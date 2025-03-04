//===-- Unittests for bitshk ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitsFxTest.h"

#include "llvm-libc-types/stdfix-types.h" // int_hk_t
#include "src/stdfix/bitshk.h"

LIST_BITSFX_TESTS(hk, short accum, int_hk_t, LIBC_NAMESPACE::bitshk);
