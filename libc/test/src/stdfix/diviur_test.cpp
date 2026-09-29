//===-- Unittests for diviur ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DiviFxTest.h"

#include "llvm-libc-macros/stdfix-macros.h"
#include "src/stdfix/diviur.h"

LIST_DIVIFX_TESTS(ur, unsigned int, unsigned fract, LIBC_NAMESPACE::diviur);
