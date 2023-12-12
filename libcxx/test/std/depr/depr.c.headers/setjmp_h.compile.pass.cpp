//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <setjmp.h>
//
// Even though <setjmp.h> is not provided by libc++, we still test that
// using it with libc++ on the search path will work.

#include "test_macros.h"

// MSVC warning C4611: interaction between '_setjmp' and C++ object destruction is non-portable
TEST_MSVC_DIAGNOSTIC_IGNORED(4611)

#include <setjmp.h>

jmp_buf jb;
ASSERT_SAME_TYPE(void, decltype(longjmp(jb, 0)));

void f() { setjmp(jb); }
