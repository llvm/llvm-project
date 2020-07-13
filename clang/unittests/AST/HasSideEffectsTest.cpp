//===- unittest/AST/HasSideEffectsTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>

using namespace clang;

namespace {
class ProcessASTAction : public clang::ASTFrontendAction {
public:
  ProcessASTAction(llvm::unique_function<void(clang::ASTContext &)> Process)
      : Process(std::move(Process)) {
    assert(this->Process);
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) {
    class Consumer : public ASTConsumer {
    public:
      Consumer(llvm::function_ref<void(ASTContext &CTx)> Process)
          : Process(Process) {}

      void HandleTranslationUnit(ASTContext &Ctx) override { Process(Ctx); }

    private:
      llvm::function_ref<void(ASTContext &CTx)> Process;
    };

    return std::make_unique<Consumer>(Process);
  }

private:
  llvm::unique_function<void(clang::ASTContext &)> Process;
};

class RunHasSideEffects
    : public RecursiveASTVisitor<RunHasSideEffects> {
public:
  RunHasSideEffects(ASTContext& Ctx)
  : Ctx(Ctx) {}

  bool VisitLambdaExpr(LambdaExpr *LE) {
    LE->HasSideEffects(Ctx);
    return true;
  }

  ASTContext& Ctx;
};
} // namespace

TEST(HasSideEffectsTest, All) {
  llvm::StringRef Code = R"cpp(
void Test() {
  int msize = 4;
  float arr[msize];
  [&arr] {};
}
  )cpp";

  ASSERT_NO_FATAL_FAILURE(
    clang::tooling::runToolOnCode(
      std::make_unique<ProcessASTAction>(
          [&](clang::ASTContext &Ctx) {
              RunHasSideEffects Visitor(Ctx);
              Visitor.TraverseAST(Ctx);
          }
      ),
      Code)
  );

}
