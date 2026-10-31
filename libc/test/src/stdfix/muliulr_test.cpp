//===-- Unittests for muliulr ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MuliFxTest.h"

#include "llvm-libc-macros/stdfix-macros.h"
#include "src/stdfix/muliulr.h"

LIST_MULIFX_TESTS(ulr, unsigned long int, unsigned long fract,
                  LIBC_NAMESPACE::muliulr);
