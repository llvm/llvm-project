//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// XFAIL: LIBCXX-WINDOWS-FIXME

// Make sure TEST_HAS_TIMESPEC_GET (defined by the test suite) and
// _LIBCPP_HAS_TIMESPEC_GET (defined by libc++) stay in sync.

#include <__config>
#include "test_macros.h"

#if defined(TEST_HAS_TIMESPEC_GET) != defined(_LIBCPP_HAS_TIMESPEC_GET)
#   error "TEST_HAS_TIMESPEC_GET and _LIBCPP_HAS_TIMESPEC_GET are out of sync"
#endif
