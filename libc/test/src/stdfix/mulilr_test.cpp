//===-- Unittests for mulilr ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MuliFxTest.h"

#include "llvm-libc-macros/stdfix-macros.h"
#include "src/stdfix/mulilr.h"

LIST_MULIFX_TESTS(lr, long int, long fract, LIBC_NAMESPACE::mulilr);
