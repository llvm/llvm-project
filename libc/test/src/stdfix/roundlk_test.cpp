//===-- Unittests for roundlk ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RoundTest.h"

#include "src/stdfix/roundlk.h"

LIST_ROUND_TESTS(long accum, LIBC_NAMESPACE::roundlk);
