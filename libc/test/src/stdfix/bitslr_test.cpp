//===-- Unittests for bitslr ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitsFxTest.h"

#include "llvm-libc-types/int_lr_t.h" // int_lr_t
#include "src/stdfix/bitslr.h"

LIST_BITSFX_TESTS(hk, long fract, int_lr_t, LIBC_NAMESPACE::bitslr);
