//===-- Unittests for expk ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpTest.h"

#include "src/stdfix/expk.h"

LIST_EXP_TESTS(k, accum, LIBC_NAMESPACE::expk);
