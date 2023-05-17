//===- unittest/Tooling/HeaderAnalysisTest.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Inclusions/HeaderAnalysis.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Testing/TestAST.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {
namespace {
using testing::Eq;

TEST(HeaderAnalysisTest, IsSelfContained) {
  TestInputs Inputs;
  Inputs.Code = R"cpp(
  #include "headerguard.h"
  #include "pragmaonce.h"
  #import "imported.h"

  #include "bad.h"
  #include "unguarded.h"
  )cpp";

  Inputs.ExtraFiles["headerguard.h"] = R"cpp(
  #ifndef HEADER_H
  #define HEADER_H

  #endif HEADER_H
  )cpp";
  Inputs.ExtraFiles["pragmaonce.h"] = R"cpp(
  #pragma once
  )cpp";
  Inputs.ExtraFiles["imported.h"] = "";

  Inputs.ExtraFiles["unguarded.h"] = "";
  Inputs.ExtraFiles["bad.h"] = R"cpp(
  #pragma once

  #if defined(INSIDE_H)
  #error "Only ... can be included directly"
  #endif
  )cpp";

  TestAST AST(Inputs);
  const auto &SM = AST.sourceManager();
  auto &FM = SM.getFileManager();
  auto &HI = AST.preprocessor().getHeaderSearchInfo();
  EXPECT_TRUE(isSelfContainedHeader(FM.getFile("headerguard.h").get(), SM, HI));
  EXPECT_TRUE(isSelfContainedHeader(FM.getFile("pragmaonce.h").get(), SM, HI));
  EXPECT_TRUE(isSelfContainedHeader(FM.getFile("imported.h").get(), SM, HI));
  EXPECT_TRUE(
      isSelfContainedHeader(SM.getFileEntryForID(SM.getMainFileID()), SM, HI));

  EXPECT_FALSE(isSelfContainedHeader(FM.getFile("unguarded.h").get(), SM, HI));
  EXPECT_FALSE(isSelfContainedHeader(FM.getFile("bad.h").get(), SM, HI));
}

TEST(HeaderAnalysisTest, CodeContainsImports) {
  EXPECT_TRUE(codeContainsImports(R"cpp(
  #include "foo.h"
  #import "NSFoo.h"

  int main() {
    foo();
  }
  )cpp"));

  EXPECT_TRUE(codeContainsImports(R"cpp(
  #include "foo.h"

  int main() {
    foo();
  }

  #import "NSFoo.h"
  )cpp"));

  EXPECT_FALSE(codeContainsImports(R"cpp(
  #include "foo.h"

  int main() {
    foo();
  }
  )cpp"));
}

TEST(HeaderAnalysisTest, ParseIWYUPragma) {
  EXPECT_THAT(parseIWYUPragma("// IWYU pragma: keep"), Eq("keep"));
  EXPECT_THAT(parseIWYUPragma("// IWYU pragma:   keep  me\netc"),
              Eq("keep  me"));
  EXPECT_THAT(parseIWYUPragma("/* IWYU pragma: keep */"), Eq("keep"));
  EXPECT_EQ(parseIWYUPragma("//  IWYU pragma: keep"), std::nullopt)
      << "Prefix is sensitive to whitespace";
  EXPECT_EQ(parseIWYUPragma("// IWYU pragma:keep"), std::nullopt)
      << "Prefix is sensitive to whitespace";
  EXPECT_EQ(parseIWYUPragma("/\n* IWYU pragma: keep */"), std::nullopt)
      << "Must start with /* or //";
}

} // namespace
} // namespace tooling
} // namespace clang
