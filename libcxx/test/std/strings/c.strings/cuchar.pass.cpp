//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// XFAIL: stdlib=libc++

// Skip this test on windows. If built on top of the MSVC runtime, the
// <cuchar> header actually does exist (although not provided by us).
// This should be removed once D97870 has landed.
// UNSUPPORTED: windows

// <cuchar>

#include <cuchar>

#include "test_macros.h"

int main(int, char**)
{

  return 0;
}
