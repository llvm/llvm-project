//===- unittests/Basic/LangOptionsTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/LangOptions.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {
TEST(LangOptsTest, CStdLang) {
  LangOptions opts;
  EXPECT_FALSE(opts.getCLangStd());
  opts.GNUMode = 0;
  opts.Digraphs = 1;
  EXPECT_EQ(opts.getCLangStd(), 199409);
  opts.C99 = 1;
  EXPECT_EQ(opts.getCLangStd(), 199901);
  opts.C11 = 1;
  EXPECT_EQ(opts.getCLangStd(), 201112);
  opts.C17 = 1;
  EXPECT_EQ(opts.getCLangStd(), 201710);
  opts.C23 = 1;
  EXPECT_EQ(opts.getCLangStd(), 202311);
  opts.C2y = 1;
  EXPECT_EQ(opts.getCLangStd(), 202400);

  EXPECT_FALSE(opts.getCPlusPlusLangStd());
}

TEST(LangOptsTest, CppStdLang) {
  LangOptions opts;
  EXPECT_FALSE(opts.getCPlusPlusLangStd());
  opts.CPlusPlus = 1;
  EXPECT_EQ(opts.getCPlusPlusLangStd(), 199711);
  opts.CPlusPlus11 = 1;
  EXPECT_EQ(opts.getCPlusPlusLangStd(), 201103);
  opts.CPlusPlus14 = 1;
  EXPECT_EQ(opts.getCPlusPlusLangStd(), 201402);
  opts.CPlusPlus17 = 1;
  EXPECT_EQ(opts.getCPlusPlusLangStd(), 201703);
  opts.CPlusPlus20 = 1;
  EXPECT_EQ(opts.getCPlusPlusLangStd(), 202002);
  opts.CPlusPlus23 = 1;
  EXPECT_EQ(opts.getCPlusPlusLangStd(), 202302);
  opts.CPlusPlus26 = 1;
  EXPECT_EQ(opts.getCPlusPlusLangStd(), 202400);

  EXPECT_FALSE(opts.getCLangStd());
}
} // namespace
