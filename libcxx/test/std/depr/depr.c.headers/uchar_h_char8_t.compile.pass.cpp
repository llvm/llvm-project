//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// The following platforms do not provide mbrtoc8 and c8rtomb so the tests fail
// XFAIL: target={{.+}}-aix{{.*}}
// XFAIL: android
// XFAIL: darwin
// XFAIL: freebsd
// XFAIL: windows
// XFAIL: glibc-no-char8_t-support
// XFAIL: LIBCXX-PICOLIBC-FIXME

// <uchar.h>

#include <uchar.h>

ASSERT_SAME_TYPE(size_t, decltype(mbrtoc8((char8_t*)0, (const char*)0, (size_t)0, (mbstate_t*)0)));
ASSERT_SAME_TYPE(size_t, decltype(c8rtomb((char*)0, (char8_t)0, (mbstate_t*)0)));
