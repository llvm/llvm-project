//===- unittests/Frontend/FrontendActionTest.cpp - FrontendAction tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
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

  bool BeginSourceFileAction(CompilerInstance &ci) override {
    if (EnableIncrementalProcessing)
      ci.getPreprocessor().enableIncrementalProcessing();

    return ASTFrontendAction::BeginSourceFileAction(ci);
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
  auto invocation = std::make_shared<CompilerInvocation>();
  invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc",
      MemoryBuffer::getMemBuffer("int main() { float x; }").release());
  invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", InputKind::CXX));
  invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance compiler;
  compiler.setInvocation(std::move(invocation));
  compiler.createDiagnostics();

  TestASTFrontendAction test_action;
  ASSERT_TRUE(compiler.ExecuteAction(test_action));
  ASSERT_EQ(2U, test_action.decl_names.size());
  EXPECT_EQ("main", test_action.decl_names[0]);
  EXPECT_EQ("x", test_action.decl_names[1]);
}

TEST(ASTFrontendAction, IncrementalParsing) {
  auto invocation = std::make_shared<CompilerInvocation>();
  invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc",
      MemoryBuffer::getMemBuffer("int main() { float x; }").release());
  invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", InputKind::CXX));
  invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance compiler;
  compiler.setInvocation(std::move(invocation));
  compiler.createDiagnostics();

  TestASTFrontendAction test_action(/*enableIncrementalProcessing=*/true);
  ASSERT_TRUE(compiler.ExecuteAction(test_action));
  ASSERT_EQ(2U, test_action.decl_names.size());
  EXPECT_EQ("main", test_action.decl_names[0]);
  EXPECT_EQ("x", test_action.decl_names[1]);
}

TEST(ASTFrontendAction, LateTemplateIncrementalParsing) {
  auto invocation = std::make_shared<CompilerInvocation>();
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
  invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", InputKind::CXX));
  invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance compiler;
  compiler.setInvocation(std::move(invocation));
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
  auto Invocation = std::make_shared<CompilerInvocation>();
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc",
      MemoryBuffer::getMemBuffer("int main() { float x; }").release());
  Invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", InputKind::CXX));
  Invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  Invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance Compiler;
  Compiler.setInvocation(std::move(Invocation));
  Compiler.createDiagnostics();

  TestPPCallbacks *Callbacks = new TestPPCallbacks;
  TestPPCallbacksFrontendAction TestAction(Callbacks);
  ASSERT_FALSE(Callbacks->SeenEnd);
  ASSERT_FALSE(TestAction.SeenEnd);
  ASSERT_TRUE(Compiler.ExecuteAction(TestAction));
  // Check that EndOfMainFile was called before EndSourceFileAction.
  ASSERT_TRUE(TestAction.SeenEnd);
}

class TypoExternalSemaSource : public ExternalSemaSource {
  CompilerInstance &CI;

public:
  TypoExternalSemaSource(CompilerInstance &CI) : CI(CI) {}

  TypoCorrection CorrectTypo(const DeclarationNameInfo &Typo, int LookupKind,
                             Scope *S, CXXScopeSpec *SS,
                             CorrectionCandidateCallback &CCC,
                             DeclContext *MemberContext, bool EnteringContext,
                             const ObjCObjectPointerType *OPT) override {
    // Generate a fake typo correction with one attached note.
    ASTContext &Ctx = CI.getASTContext();
    TypoCorrection TC(DeclarationName(&Ctx.Idents.get("moo")));
    unsigned DiagID = Ctx.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Note, "This is a note");
    TC.addExtraDiagnostic(PartialDiagnostic(DiagID, Ctx.getDiagAllocator()));
    return TC;
  }
};

struct TypoDiagnosticConsumer : public DiagnosticConsumer {
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override {
    // Capture errors and notes. There should be one of each.
    if (DiagLevel == DiagnosticsEngine::Error) {
      assert(Error.empty());
      Info.FormatDiagnostic(Error);
    } else {
      assert(Note.empty());
      Info.FormatDiagnostic(Note);
    }
  }
  SmallString<32> Error;
  SmallString<32> Note;
};

TEST(ASTFrontendAction, ExternalSemaSource) {
  auto Invocation = std::make_shared<CompilerInvocation>();
  Invocation->getLangOpts()->CPlusPlus = true;
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc", MemoryBuffer::getMemBuffer("void fooo();\n"
                                            "int main() { foo(); }")
                     .release());
  Invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", InputKind::CXX));
  Invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  Invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance Compiler;
  Compiler.setInvocation(std::move(Invocation));
  auto *TDC = new TypoDiagnosticConsumer;
  Compiler.createDiagnostics(TDC, /*ShouldOwnClient=*/true);
  Compiler.setExternalSemaSource(new TypoExternalSemaSource(Compiler));

  SyntaxOnlyAction TestAction;
  ASSERT_TRUE(Compiler.ExecuteAction(TestAction));
  // There should be one error correcting to 'moo' and a note attached to it.
  EXPECT_EQ("use of undeclared identifier 'foo'; did you mean 'moo'?",
            TDC->Error.str().str());
  EXPECT_EQ("This is a note", TDC->Note.str().str());
}

} // anonymous namespace
