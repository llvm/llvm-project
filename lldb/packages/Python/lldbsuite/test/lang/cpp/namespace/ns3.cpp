//===-- ns3.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ns.h"
extern int func();

// Note: the following function must be before the using.
void test_lookup_before_using_directive()
{
  // BP_before_using_directive
  std::printf("before using directive: func() = %d\n", func()); // eval func(), exp: 1
}
using namespace A;
void test_lookup_after_using_directive()
{
  // BP_after_using_directive
  //printf("func() = %d\n", func()); // eval func(), exp: error, amiguous
  std::printf("after using directive: func2() = %d\n", func2()); // eval func2(), exp: 3
  std::printf("after using directive: ::func() = %d\n", ::func()); // eval ::func(), exp: 1
  std::printf("after using directive: B::func() = %d\n", B::func()); // eval B::func(), exp: 4
}
