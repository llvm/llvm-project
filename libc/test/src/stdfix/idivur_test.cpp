//===-- Unittests for idivur ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IdivTest.h"

#include "llvm-libc-macros/stdfix-macros.h" // unsigned fract
#include "src/stdfix/idivur.h"

LIST_IDIV_TESTS(ur, unsigned fract, unsigned int, LIBC_NAMESPACE::idivur);
