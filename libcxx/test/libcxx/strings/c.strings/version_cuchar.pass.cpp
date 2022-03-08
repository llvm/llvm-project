//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// The system-provided <uchar.h> seems to be broken on AIX
// XFAIL: LIBCXX-AIX-FIXME

// <cuchar>

#include <cuchar>

#include "test_macros.h"

#ifndef _LIBCPP_VERSION
#error _LIBCPP_VERSION not defined
#endif

int main(int, char**)
{

  return 0;
}
