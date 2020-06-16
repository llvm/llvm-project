//===- InternalNames.cpp - InternalNames unit tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/InternalNames.h"
#include "gtest/gtest.h"

using namespace fir;
using namespace llvm;

TEST(genericName, MyTest) {
  NameUniquer obj;
  std::string val = obj.doCommonBlock("hello");
  std::string val2 = "_QBhello";
  EXPECT_EQ(val, val2);
}
