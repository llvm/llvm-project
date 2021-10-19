//===-- Unittests for lldiv -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DivTest.h"

#include "src/stdlib/lldiv.h"

#include <stdlib.h>

LIST_DIV_TESTS(long long, lldiv_t, __llvm_libc::lldiv)
