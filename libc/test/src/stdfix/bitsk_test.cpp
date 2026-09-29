//===-- Unittests for bitsk -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitsFxTest.h"

#include "llvm-libc-types/int_k_t.h" // int_k_t
#include "src/stdfix/bitsk.h"

LIST_BITSFX_TESTS(k, accum, int_k_t, LIBC_NAMESPACE::bitsk);
