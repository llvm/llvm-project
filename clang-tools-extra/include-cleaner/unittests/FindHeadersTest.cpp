//===--- FindHeadersTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang-include-cleaner/Record.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Testing/TestAST.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::include_cleaner {
namespace {
using testing::UnorderedElementsAre;

TEST(FindIncludeHeaders, IWYU) {
  TestInputs Inputs;
  PragmaIncludes PI;
  Inputs.MakeAction = [&PI] {
    struct Hook : public PreprocessOnlyAction {
    public:
      Hook(PragmaIncludes *Out) : Out(Out) {}
      bool BeginSourceFileAction(clang::CompilerInstance &CI) override {
        Out->record(CI);
        return true;
      }

      PragmaIncludes *Out;
    };
    return std::make_unique<Hook>(&PI);
  };

  Inputs.Code = R"cpp(
    #include "header1.h"
    #include "header2.h"
  )cpp";
  Inputs.ExtraFiles["header1.h"] = R"cpp(
    // IWYU pragma: private, include "path/public.h"
  )cpp";
  Inputs.ExtraFiles["header2.h"] = R"cpp(
    #include "detail1.h" // IWYU pragma: export

    // IWYU pragma: begin_exports
    #include "detail2.h"
    // IWYU pragma: end_exports

    #include "normal.h"
  )cpp";
  Inputs.ExtraFiles["normal.h"] = Inputs.ExtraFiles["detail1.h"] =
      Inputs.ExtraFiles["detail2.h"] = "";
  TestAST AST(Inputs);
  const auto &SM = AST.sourceManager();
  auto &FM = SM.getFileManager();
  // Returns the source location for the start of the file.
  auto SourceLocFromFile = [&](llvm::StringRef FileName) {
    return SM.translateFileLineCol(FM.getFile(FileName).get(),
                                   /*Line=*/1, /*Col=*/1);
  };

  EXPECT_THAT(findHeaders(SourceLocFromFile("header1.h"), SM, PI),
              UnorderedElementsAre(Header("\"path/public.h\"")));

  EXPECT_THAT(findHeaders(SourceLocFromFile("detail1.h"), SM, PI),
              UnorderedElementsAre(Header(FM.getFile("header2.h").get()),
                                   Header(FM.getFile("detail1.h").get())));
  EXPECT_THAT(findHeaders(SourceLocFromFile("detail2.h"), SM, PI),
              UnorderedElementsAre(Header(FM.getFile("header2.h").get()),
                                   Header(FM.getFile("detail2.h").get())));

  EXPECT_THAT(findHeaders(SourceLocFromFile("normal.h"), SM, PI),
              UnorderedElementsAre(Header(FM.getFile("normal.h").get())));
}

} // namespace
} // namespace clang::include_cleaner