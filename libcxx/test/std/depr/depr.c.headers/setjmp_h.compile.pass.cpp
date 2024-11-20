//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MSVC warning C4611: interaction between '_setjmp' and C++ object destruction is non-portable
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd4611

// test <setjmp.h>
//
// Even though <setjmp.h> is not provided by libc++, we still test that
// using it with libc++ on the search path will work.

#include <setjmp.h>

#include "test_macros.h"

jmp_buf jb;
ASSERT_SAME_TYPE(void, decltype(longjmp(jb, 0)));

void f() { setjmp(jb); }
