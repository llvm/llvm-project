//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <thread>

// #define __STDCPP_THREADS__ __cplusplus

#include <thread>

#include "test_macros.h"

int main(int, char**)
{
#ifndef __STDCPP_THREADS__
#error __STDCPP_THREADS__ is not defined
#endif

  return 0;
}
