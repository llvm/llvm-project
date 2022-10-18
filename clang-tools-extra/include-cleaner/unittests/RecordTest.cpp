//===-- RecordTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Record.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Testing/TestAST.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::include_cleaner {
namespace {

// Matches a Decl* if it is a NamedDecl with the given name.
MATCHER_P(Named, N, "") {
  if (const NamedDecl *ND = llvm::dyn_cast<NamedDecl>(arg)) {
    if (N == ND->getNameAsString())
      return true;
  }
  std::string S;
  llvm::raw_string_ostream OS(S);
  arg->dump(OS);
  *result_listener << S;
  return false;
}

class RecordASTTest : public ::testing::Test {
protected:
  TestInputs Inputs;
  RecordedAST Recorded;

  RecordASTTest() {
    struct RecordAction : public ASTFrontendAction {
      RecordedAST &Out;
      RecordAction(RecordedAST &Out) : Out(Out) {}
      std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                     StringRef) override {
        return Out.record();
      }
    };
    Inputs.MakeAction = [this] {
      return std::make_unique<RecordAction>(Recorded);
    };
  }

  TestAST build() { return TestAST(Inputs); }
};

// Top-level decl from the main file is a root, nested ones aren't.
TEST_F(RecordASTTest, Namespace) {
  Inputs.Code =
      R"cpp(
      namespace ns {
        int x;
        namespace {
          int y;
        }
      }
    )cpp";
  auto AST = build();
  EXPECT_THAT(Recorded.Roots, testing::ElementsAre(Named("ns")));
}

// Decl in included file is not a root.
TEST_F(RecordASTTest, Inclusion) {
  Inputs.ExtraFiles["header.h"] = "void headerFunc();";
  Inputs.Code = R"cpp(
    #include "header.h"
    void mainFunc();
  )cpp";
  auto AST = build();
  EXPECT_THAT(Recorded.Roots, testing::ElementsAre(Named("mainFunc")));
}

// Decl from macro expanded into the main file is a root.
TEST_F(RecordASTTest, Macros) {
  Inputs.ExtraFiles["header.h"] = "#define X void x();";
  Inputs.Code = R"cpp(
    #include "header.h"
    X
  )cpp";
  auto AST = build();
  EXPECT_THAT(Recorded.Roots, testing::ElementsAre(Named("x")));
}

} // namespace
} // namespace clang::include_cleaner
