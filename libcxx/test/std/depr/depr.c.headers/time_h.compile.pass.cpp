//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <time.h>

#include <time.h>

#include "test_macros.h"

#ifndef NULL
#error NULL not defined
#endif

#ifndef CLOCKS_PER_SEC
#error CLOCKS_PER_SEC not defined
#endif

clock_t c = 0;
size_t s = 0;
time_t t = 0;
tm tmv = {};
char* c1 = 0;
const char* c2 = 0;
ASSERT_SAME_TYPE(clock_t, decltype(clock()));
ASSERT_SAME_TYPE(double,  decltype(difftime(t, t)));
ASSERT_SAME_TYPE(time_t,  decltype(mktime(&tmv)));
ASSERT_SAME_TYPE(time_t,  decltype(time(&t)));
ASSERT_SAME_TYPE(char*,   decltype(asctime(&tmv)));
ASSERT_SAME_TYPE(char*,   decltype(ctime(&t)));
ASSERT_SAME_TYPE(tm*,     decltype(gmtime(&t)));
ASSERT_SAME_TYPE(tm*,     decltype(localtime(&t)));
ASSERT_SAME_TYPE(size_t,  decltype(strftime(c1, s, c2, &tmv)));
