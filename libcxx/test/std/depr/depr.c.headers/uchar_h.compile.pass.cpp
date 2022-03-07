//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Apple platforms don't provide <uchar.h> yet, so these tests fail.
// XFAIL: target={{.+}}-apple-{{.+}}

// The system-provided <uchar.h> seems to be broken on AIX
// XFAIL: LIBCXX-AIX-FIXME

// <uchar.h>

#include <uchar.h>

#include "test_macros.h"

// __STDC_UTF_16__ may or may not be defined by the C standard library
// __STDC_UTF_32__ may or may not be defined by the C standard library

ASSERT_SAME_TYPE(size_t, decltype(mbrtoc16((char16_t*)0, (const char*)0, (size_t)0, (mbstate_t*)0)));
ASSERT_SAME_TYPE(size_t, decltype(c16rtomb((char*)0, (char16_t)0, (mbstate_t*)0)));

ASSERT_SAME_TYPE(size_t, decltype(mbrtoc32((char32_t*)0, (const char*)0, (size_t)0, (mbstate_t*)0)));
ASSERT_SAME_TYPE(size_t, decltype(c16rtomb((char*)0, (char32_t)0, (mbstate_t*)0)));
