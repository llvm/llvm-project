//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for muliuk.
///
//===----------------------------------------------------------------------===//

#include "test/src/stdfix/MuliFxTest.h"

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/stdfix/muliuk.h"

LIST_MULIFX_TESTS(uk, unsigned int, unsigned accum, LIBC_NAMESPACE::muliuk);
