//===- unittests/Analysis/ThreadSafetyTest.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/ThreadSafety.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace threadSafety;

namespace {

TEST(ThreadSafetyTest, LeaveFunctionOnEarlyExit) {
  auto Built = tooling::buildASTFromCodeWithArgs(
      R"cpp(
        struct __attribute__((capability("mutex"))) Mu {
          void Lock() __attribute__((acquire_capability())) {}
          bool TryLock() __attribute__((try_acquire_capability(true))) {
            return true;
          }
        };
      )cpp",
      {"-std=c++17"});
  ASSERT_TRUE(Built);
  ASTContext &C = Built->getASTContext();

  struct CountingHandler : ThreadSafetyHandler {
    unsigned Entered = 0;
    unsigned Left = 0;
    void enterFunction(const FunctionDecl *) override { ++Entered; }
    void leaveFunction(const FunctionDecl *) override { ++Left; }
  } Handler;

  AnalysisDeclContextManager ADCM(C);
  BeforeSet *Cache = nullptr;
  for (const Decl *D : C.getTranslationUnitDecl()->decls()) {
    if (const auto *RD = dyn_cast<CXXRecordDecl>(D);
        RD && RD->getName() == "Mu") {
      for (const CXXMethodDecl *MD : RD->methods())
        if (MD->getName() == "Lock" || MD->getName() == "TryLock")
          runThreadSafetyAnalysis(*ADCM.getContext(MD), Handler, &Cache);
    }
  }
  threadSafetyCleanup(Cache);
  EXPECT_EQ(Handler.Entered, 2u);
  EXPECT_EQ(Handler.Left, 2u);
}

} // namespace
