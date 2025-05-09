//===-- Unittests for bitsuhr ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitsFxTest.h"

#include "llvm-libc-types/stdfix-types.h" // uint_uhr_t
#include "src/stdfix/bitsuhr.h"

LIST_BITSFX_TESTS(uhr, unsigned short fract, uint_uhr_t,
                  LIBC_NAMESPACE::bitsuhr);
