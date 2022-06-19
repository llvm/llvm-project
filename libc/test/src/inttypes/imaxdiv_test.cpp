//===-- Unittests for imaxdiv ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../stdlib/DivTest.h"

#include "src/inttypes/imaxdiv.h"

#include <inttypes.h>

LIST_DIV_TESTS(intmax_t, imaxdiv_t, __llvm_libc::imaxdiv)
