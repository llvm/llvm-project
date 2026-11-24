//===-- Unittests for divik -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/src/stdfix/DiviFxTest.h"

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/stdfix/divik.h"

LIST_DIVIFX_TESTS(k, int, accum, LIBC_NAMESPACE::divik);
