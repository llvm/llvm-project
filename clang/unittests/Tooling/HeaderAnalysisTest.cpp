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
#include "gtest/gtest.h"

namespace clang {
namespace tooling {
namespace {

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
  auto getFileID = [&](llvm::StringRef FileName) {
    return SM.translateFile(FM.getFile(FileName).get());
  };
  EXPECT_TRUE(isSelfContainedHeader(getFileID("headerguard.h"), SM, HI));
  EXPECT_TRUE(isSelfContainedHeader(getFileID("pragmaonce.h"), SM, HI));
  EXPECT_TRUE(isSelfContainedHeader(getFileID("imported.h"), SM, HI));

  EXPECT_FALSE(isSelfContainedHeader(getFileID("unguarded.h"), SM, HI));
  EXPECT_FALSE(isSelfContainedHeader(getFileID("bad.h"), SM, HI));
}

} // namespace
} // namespace tooling
} // namespace clang
