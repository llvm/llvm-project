//===- unittest/AST/ExternalASTSourceTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for Clang's ExternalASTSource.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace llvm;


class TestFrontendAction : public ASTFrontendAction {
public:
  TestFrontendAction(ExternalASTSource *Source) : Source(Source) {}

private:
  void ExecuteAction() override {
    getCompilerInstance().getASTContext().setExternalSource(Source);
    getCompilerInstance().getASTContext().getTranslationUnitDecl()
        ->setHasExternalVisibleStorage();
    return ASTFrontendAction::ExecuteAction();
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<ASTConsumer>();
  }

  IntrusiveRefCntPtr<ExternalASTSource> Source;
};

bool testExternalASTSource(ExternalASTSource *Source,
                           StringRef FileContents) {
  CompilerInstance Compiler;
  Compiler.createDiagnostics(*llvm::vfs::getRealFileSystem());

  auto Invocation = std::make_shared<CompilerInvocation>();
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc", MemoryBuffer::getMemBuffer(FileContents).release());
  const char *Args[] = { "test.cc" };
  CompilerInvocation::CreateFromArgs(*Invocation, Args,
                                     Compiler.getDiagnostics());
  Compiler.setInvocation(std::move(Invocation));

  TestFrontendAction Action(Source);
  return Compiler.ExecuteAction(Action);
}


// Ensure that a failed name lookup into an external source only occurs once.
TEST(ExternalASTSourceTest, FailedLookupOccursOnce) {
  struct TestSource : ExternalASTSource {
    TestSource(unsigned &Calls) : Calls(Calls) {}

    bool
    FindExternalVisibleDeclsByName(const DeclContext *, DeclarationName Name,
                                   const DeclContext *OriginalDC) override {
      if (Name.getAsString() == "j")
        ++Calls;
      return false;
    }

    unsigned &Calls;
  };

  unsigned Calls = 0;
  ASSERT_TRUE(testExternalASTSource(new TestSource(Calls), "int j, k = j;"));
  EXPECT_EQ(1u, Calls);
}
