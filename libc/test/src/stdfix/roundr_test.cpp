//===-- Unittests for roundr ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RoundTest.h"

#include "src/stdfix/roundr.h"

LIST_ROUND_TESTS(fract, LIBC_NAMESPACE::roundr);
