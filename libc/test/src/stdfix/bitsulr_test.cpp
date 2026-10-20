//===-- Unittests for bitsulr ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitsFxTest.h"

#include "llvm-libc-types/uint_ulr_t.h" // uint_ulr_t
#include "src/stdfix/bitsulr.h"

LIST_BITSFX_TESTS(ulr, unsigned long fract, uint_ulr_t,
                  LIBC_NAMESPACE::bitsulr);
