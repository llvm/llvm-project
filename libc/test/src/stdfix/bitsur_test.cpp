//===-- Unittests for bitsur ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitsFxTest.h"

#include "llvm-libc-types/stdfix-types.h" // uint_ur_t
#include "src/stdfix/bitsur.h"

LIST_BITSFX_TESTS(ur, unsigned fract, uint_ur_t, LIBC_NAMESPACE::bitsur);
