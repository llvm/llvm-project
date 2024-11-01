//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <setjmp.h>

#include <setjmp.h>

#include "test_macros.h"

#ifndef setjmp
#error setjmp not defined
#endif

jmp_buf jb;
ASSERT_SAME_TYPE(void, decltype(longjmp(jb, 0)));
