//===- unittest/AST/ExternalASTSourceTest.cpp -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for Clang's ExternalASTSource.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PreprocessorOptions.h"
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
    return llvm::make_unique<ASTConsumer>();
  }

  IntrusiveRefCntPtr<ExternalASTSource> Source;
};

bool testExternalASTSource(ExternalASTSource *Source,
                           StringRef FileContents) {
  CompilerInstance Compiler;
  Compiler.createDiagnostics();

  CompilerInvocation *Invocation = new CompilerInvocation;
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc", MemoryBuffer::getMemBuffer(FileContents).release());
  const char *Args[] = { "test.cc" };
  CompilerInvocation::CreateFromArgs(*Invocation, Args,
                                     Args + array_lengthof(Args),
                                     Compiler.getDiagnostics());
  Compiler.setInvocation(Invocation);

  TestFrontendAction Action(Source);
  return Compiler.ExecuteAction(Action);
}


// Ensure that a failed name lookup into an external source only occurs once.
TEST(ExternalASTSourceTest, FailedLookupOccursOnce) {
  struct TestSource : ExternalASTSource {
    TestSource(unsigned &Calls) : Calls(Calls) {}

    bool FindExternalVisibleDeclsByName(const DeclContext *,
                                        DeclarationName Name) override {
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
