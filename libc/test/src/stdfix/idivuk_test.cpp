//===-- Unittests for idivuk ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IdivTest.h"

#include "llvm-libc-macros/stdfix-macros.h" // unsigned accum
#include "src/stdfix/idivuk.h"

LIST_IDIV_TESTS(uk, unsigned accum, unsigned int, LIBC_NAMESPACE::idivuk);
