//===- unittests/Frontend/FrontendActionTest.cpp - FrontendAction tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class TestASTFrontendAction : public ASTFrontendAction {
public:
  TestASTFrontendAction(bool enableIncrementalProcessing = false,
                        bool actOnEndOfTranslationUnit = false)
    : EnableIncrementalProcessing(enableIncrementalProcessing),
      ActOnEndOfTranslationUnit(actOnEndOfTranslationUnit) { }

  bool EnableIncrementalProcessing;
  bool ActOnEndOfTranslationUnit;
  std::vector<std::string> decl_names;

  bool BeginSourceFileAction(CompilerInstance &ci,
                             StringRef filename) override {
    if (EnableIncrementalProcessing)
      ci.getPreprocessor().enableIncrementalProcessing();

    return ASTFrontendAction::BeginSourceFileAction(ci, filename);
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return llvm::make_unique<Visitor>(CI, ActOnEndOfTranslationUnit,
                                      decl_names);
  }

private:
  class Visitor : public ASTConsumer, public RecursiveASTVisitor<Visitor> {
  public:
    Visitor(CompilerInstance &CI, bool ActOnEndOfTranslationUnit,
            std::vector<std::string> &decl_names) :
      CI(CI), ActOnEndOfTranslationUnit(ActOnEndOfTranslationUnit),
      decl_names_(decl_names) {}

    void HandleTranslationUnit(ASTContext &context) override {
      if (ActOnEndOfTranslationUnit) {
        CI.getSema().ActOnEndOfTranslationUnit();
      }
      TraverseDecl(context.getTranslationUnitDecl());
    }

    virtual bool VisitNamedDecl(NamedDecl *Decl) {
      decl_names_.push_back(Decl->getQualifiedNameAsString());
      return true;
    }

  private:
    CompilerInstance &CI;
    bool ActOnEndOfTranslationUnit;
    std::vector<std::string> &decl_names_;
  };
};

TEST(ASTFrontendAction, Sanity) {
  CompilerInvocation *invocation = new CompilerInvocation;
  invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc",
      MemoryBuffer::getMemBuffer("int main() { float x; }").release());
  invocation->getFrontendOpts().Inputs.push_back(FrontendInputFile("test.cc",
                                                                   IK_CXX));
  invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance compiler;
  compiler.setInvocation(invocation);
  compiler.createDiagnostics();

  TestASTFrontendAction test_action;
  ASSERT_TRUE(compiler.ExecuteAction(test_action));
  ASSERT_EQ(2U, test_action.decl_names.size());
  EXPECT_EQ("main", test_action.decl_names[0]);
  EXPECT_EQ("x", test_action.decl_names[1]);
}

TEST(ASTFrontendAction, IncrementalParsing) {
  CompilerInvocation *invocation = new CompilerInvocation;
  invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc",
      MemoryBuffer::getMemBuffer("int main() { float x; }").release());
  invocation->getFrontendOpts().Inputs.push_back(FrontendInputFile("test.cc",
                                                                   IK_CXX));
  invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance compiler;
  compiler.setInvocation(invocation);
  compiler.createDiagnostics();

  TestASTFrontendAction test_action(/*enableIncrementalProcessing=*/true);
  ASSERT_TRUE(compiler.ExecuteAction(test_action));
  ASSERT_EQ(2U, test_action.decl_names.size());
  EXPECT_EQ("main", test_action.decl_names[0]);
  EXPECT_EQ("x", test_action.decl_names[1]);
}

TEST(ASTFrontendAction, LateTemplateIncrementalParsing) {
  CompilerInvocation *invocation = new CompilerInvocation;
  invocation->getLangOpts()->CPlusPlus = true;
  invocation->getLangOpts()->DelayedTemplateParsing = true;
  invocation->getPreprocessorOpts().addRemappedFile(
    "test.cc", MemoryBuffer::getMemBuffer(
      "template<typename T> struct A { A(T); T data; };\n"
      "template<typename T> struct B: public A<T> {\n"
      "  B();\n"
      "  B(B const& b): A<T>(b.data) {}\n"
      "};\n"
      "B<char> c() { return B<char>(); }\n").release());
  invocation->getFrontendOpts().Inputs.push_back(FrontendInputFile("test.cc",
                                                                   IK_CXX));
  invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance compiler;
  compiler.setInvocation(invocation);
  compiler.createDiagnostics();

  TestASTFrontendAction test_action(/*enableIncrementalProcessing=*/true,
                                    /*actOnEndOfTranslationUnit=*/true);
  ASSERT_TRUE(compiler.ExecuteAction(test_action));
  ASSERT_EQ(13U, test_action.decl_names.size());
  EXPECT_EQ("A", test_action.decl_names[0]);
  EXPECT_EQ("c", test_action.decl_names[12]);
}

struct TestPPCallbacks : public PPCallbacks {
  TestPPCallbacks() : SeenEnd(false) {}

  void EndOfMainFile() override { SeenEnd = true; }

  bool SeenEnd;
};

class TestPPCallbacksFrontendAction : public PreprocessorFrontendAction {
  TestPPCallbacks *Callbacks;

public:
  TestPPCallbacksFrontendAction(TestPPCallbacks *C)
      : Callbacks(C), SeenEnd(false) {}

  void ExecuteAction() override {
    Preprocessor &PP = getCompilerInstance().getPreprocessor();
    PP.addPPCallbacks(std::unique_ptr<TestPPCallbacks>(Callbacks));
    PP.EnterMainSourceFile();
  }
  void EndSourceFileAction() override { SeenEnd = Callbacks->SeenEnd; }

  bool SeenEnd;
};

TEST(PreprocessorFrontendAction, EndSourceFile) {
  CompilerInvocation *Invocation = new CompilerInvocation;
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc",
      MemoryBuffer::getMemBuffer("int main() { float x; }").release());
  Invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", IK_CXX));
  Invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  Invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance Compiler;
  Compiler.setInvocation(Invocation);
  Compiler.createDiagnostics();

  TestPPCallbacks *Callbacks = new TestPPCallbacks;
  TestPPCallbacksFrontendAction TestAction(Callbacks);
  ASSERT_FALSE(Callbacks->SeenEnd);
  ASSERT_FALSE(TestAction.SeenEnd);
  ASSERT_TRUE(Compiler.ExecuteAction(TestAction));
  // Check that EndOfMainFile was called before EndSourceFileAction.
  ASSERT_TRUE(TestAction.SeenEnd);
}

} // anonymous namespace
