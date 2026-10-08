//===-- Unittests for diviulk ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DiviFxTest.h"

#include "llvm-libc-macros/stdfix-macros.h"
#include "src/stdfix/diviulk.h"

LIST_DIVIFX_TESTS(ulk, unsigned long int, unsigned long accum,
                  LIBC_NAMESPACE::diviulk);
