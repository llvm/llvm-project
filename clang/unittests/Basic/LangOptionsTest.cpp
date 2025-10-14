//===- unittests/Basic/LangOptionsTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/LangOptions.h"
#include "clang/Testing/CommandLineArgs.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {
TEST(LangOptsTest, CStdLang) {
  LangOptions opts;
  EXPECT_FALSE(opts.GetCLangStd());
  opts.C99 = 1;
  EXPECT_EQ(opts.GetCLangStd(), clang::LangOptionsBase::CLangStd::C_99);
  opts.C11 = 1;
  EXPECT_EQ(opts.GetCLangStd(), clang::LangOptionsBase::CLangStd::C_11);
  opts.C17 = 1;
  EXPECT_EQ(opts.GetCLangStd(), clang::LangOptionsBase::CLangStd::C_17);
  opts.C23 = 1;
  EXPECT_EQ(opts.GetCLangStd(), clang::LangOptionsBase::CLangStd::C_23);
  opts.C2y = 1;
  EXPECT_EQ(opts.GetCLangStd(), clang::LangOptionsBase::CLangStd::C_2y);

  EXPECT_FALSE(opts.GetCPlusPlusLangStd());
}

TEST(LangOptsTest, CppStdLang) {
  LangOptions opts;
  EXPECT_FALSE(opts.GetCPlusPlusLangStd());
  opts.CPlusPlus = 1;
  EXPECT_EQ(opts.GetCPlusPlusLangStd(),
            clang::LangOptionsBase::CPlusPlusLangStd::CPP_03);
  opts.CPlusPlus11 = 1;
  EXPECT_EQ(opts.GetCPlusPlusLangStd(),
            clang::LangOptionsBase::CPlusPlusLangStd::CPP_11);
  opts.CPlusPlus14 = 1;
  EXPECT_EQ(opts.GetCPlusPlusLangStd(),
            clang::LangOptionsBase::CPlusPlusLangStd::CPP_14);
  opts.CPlusPlus17 = 1;
  EXPECT_EQ(opts.GetCPlusPlusLangStd(),
            clang::LangOptionsBase::CPlusPlusLangStd::CPP_17);
  opts.CPlusPlus20 = 1;
  EXPECT_EQ(opts.GetCPlusPlusLangStd(),
            clang::LangOptionsBase::CPlusPlusLangStd::CPP_20);
  opts.CPlusPlus23 = 1;
  EXPECT_EQ(opts.GetCPlusPlusLangStd(),
            clang::LangOptionsBase::CPlusPlusLangStd::CPP_23);
  opts.CPlusPlus26 = 1;
  EXPECT_EQ(opts.GetCPlusPlusLangStd(),
            clang::LangOptionsBase::CPlusPlusLangStd::CPP_26);

  EXPECT_FALSE(opts.GetCLangStd());
}
} // namespace
