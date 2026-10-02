//===-- Unittests for diviulr ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DiviFxTest.h"

#include "llvm-libc-macros/stdfix-macros.h"
#include "src/stdfix/diviulr.h"

LIST_DIVIFX_TESTS(ulr, unsigned long int, unsigned long fract,
                  LIBC_NAMESPACE::diviulr);
