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

// mbrtoc16 not defined.
// XFAIL: LIBCXX-PICOLIBC-FIXME

// <cuchar>

#include <cuchar>

#include "test_macros.h"

// __STDC_UTF_16__ may or may not be defined by the C standard library
// __STDC_UTF_32__ may or may not be defined by the C standard library

#if !defined(TEST_HAS_NO_C8RTOMB_MBRTOC8)
ASSERT_SAME_TYPE(std::size_t, decltype(std::mbrtoc8((char8_t*)0, (const char*)0, (size_t)0, (mbstate_t*)0)));
ASSERT_SAME_TYPE(std::size_t, decltype(std::c8rtomb((char*)0, (char8_t)0, (mbstate_t*)0)));
#endif

ASSERT_SAME_TYPE(std::size_t, decltype(std::mbrtoc16((char16_t*)0, (const char*)0, (size_t)0, (mbstate_t*)0)));
ASSERT_SAME_TYPE(std::size_t, decltype(std::c16rtomb((char*)0, (char16_t)0, (mbstate_t*)0)));

ASSERT_SAME_TYPE(std::size_t, decltype(std::mbrtoc32((char32_t*)0, (const char*)0, (size_t)0, (mbstate_t*)0)));
ASSERT_SAME_TYPE(std::size_t, decltype(std::c16rtomb((char*)0, (char32_t)0, (mbstate_t*)0)));
