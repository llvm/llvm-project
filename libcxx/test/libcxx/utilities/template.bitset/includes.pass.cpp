//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test that <bitset> includes <string>, <stdexcept> and <iosfwd>

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <bitset>

#include "test_macros.h"

#ifndef _LIBCPP_STRING
#error <string> has not been included
#endif

#ifndef _LIBCPP_STDEXCEPT
#error <stdexcept> has not been included
#endif

#ifndef _LIBCPP_IOSFWD
#error <iosfwd> has not been included
#endif

int main(int, char**)
{

  return 0;
}
