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
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace llvm;

struct TestExternalASTSource : public ExternalASTSource {
  virtual void setupTestAST(ASTContext &Ctx) {
    Ctx.getTranslationUnitDecl()->setHasExternalVisibleStorage();
  }
};

class TestFrontendAction : public ASTFrontendAction {
public:
  TestFrontendAction(IntrusiveRefCntPtr<TestExternalASTSource> Source)
      : Source(std::move(Source)) {}

private:
  void ExecuteAction() override {
    ASTContext &Ctx = getCompilerInstance().getASTContext();
    Ctx.setExternalSource(Source);
    Source->setupTestAST(Ctx);
    return ASTFrontendAction::ExecuteAction();
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<ASTConsumer>();
  }

  IntrusiveRefCntPtr<TestExternalASTSource> Source;
};

bool testExternalASTSource(
    llvm::IntrusiveRefCntPtr<TestExternalASTSource> Source,
    StringRef FileContents) {

  auto Invocation = std::make_shared<CompilerInvocation>();
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc", MemoryBuffer::getMemBuffer(FileContents).release());
  const char *Args[] = { "test.cc" };

  DiagnosticOptions InvocationDiagOpts;
  auto InvocationDiags = CompilerInstance::createDiagnostics(
      *llvm::vfs::getRealFileSystem(), InvocationDiagOpts);
  CompilerInvocation::CreateFromArgs(*Invocation, Args, *InvocationDiags);

  CompilerInstance Compiler(std::move(Invocation));
  Compiler.setVirtualFileSystem(llvm::vfs::getRealFileSystem());
  Compiler.createDiagnostics();

  TestFrontendAction Action(Source);
  return Compiler.ExecuteAction(Action);
}

// Ensure that a failed name lookup into an external source only occurs once.
TEST(ExternalASTSourceTest, FailedLookupOccursOnce) {
  struct TestSource : TestExternalASTSource {
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
  ASSERT_TRUE(testExternalASTSource(
      llvm::makeIntrusiveRefCnt<TestSource>(Calls), "int j, k = j;"));
  EXPECT_EQ(1u, Calls);
}

namespace {

/// An external source which announces, without definitions,
///
///   template <typename T> struct A;      // primary pattern
///   template <typename T> struct A<T *>; // partial specialization pattern
///
/// and supplies the definition of whichever pattern it is asked to complete.
struct LazyTemplatePatterns : TestExternalASTSource {
  CXXRecordDecl *Primary = nullptr;
  ClassTemplatePartialSpecializationDecl *Partial = nullptr;
  unsigned PrimaryCompletions = 0;
  unsigned PartialCompletions = 0;

  void setupTestAST(ASTContext &Ctx) override {
    TranslationUnitDecl *TU = Ctx.getTranslationUnitDecl();
    IdentifierInfo &AName = Ctx.Idents.get("A");

    auto MakeParams = [&] {
      auto *Param = TemplateTypeParmDecl::Create(
          Ctx, TU, {}, {}, /*D=*/0, /*P=*/0, &Ctx.Idents.get("T"),
          /*Typename=*/true, /*ParameterPack=*/false);
      return TemplateParameterList::Create(Ctx, {}, {}, {Param}, {}, nullptr);
    };

    // template <typename T> struct A;
    Primary = CXXRecordDecl::Create(Ctx, TagDecl::TagKind::Struct, TU, {}, {},
                                    &AName);
    auto *Template = ClassTemplateDecl::Create(
        Ctx, TU, {}, DeclarationName(&AName), MakeParams(), Primary);
    Primary->setDescribedClassTemplate(Template);
    Primary->setHasExternalLexicalStorage();
    TU->addDecl(Template);

    // template <typename T> struct A<T *>;
    TemplateParameterList *PartialParams = MakeParams();
    TemplateArgument Arg(Ctx.getPointerType(Ctx.getTemplateTypeParmType(
        /*Depth=*/0, /*Index=*/0, /*ParameterPack=*/false,
        cast<TemplateTypeParmDecl>(PartialParams->getParam(0)))));
    Partial = ClassTemplatePartialSpecializationDecl::Create(
        Ctx, TagDecl::TagKind::Struct, TU, {}, {}, PartialParams, Template, Arg,
        Ctx.getCanonicalType(Ctx.getTemplateSpecializationType(
            ElaboratedTypeKeyword::Struct, TemplateName(Template), Arg, {})),
        nullptr);
    Partial->setHasExternalLexicalStorage();

    // Deduction against a partial specialization reads its arguments as
    // written, so they must be supplied even though nothing was written.
    TemplateArgumentListInfo ArgsInfo;
    ArgsInfo.addArgument(TemplateArgumentLoc(
        Arg, Ctx.getTrivialTypeSourceInfo(Arg.getAsType())));
    Partial->setTemplateArgsAsWritten(
        ASTTemplateArgumentListInfo::Create(Ctx, ArgsInfo));

    TU->addDecl(Partial);
    Template->AddPartialSpecialization(Partial, nullptr);
  }

  void CompleteType(TagDecl *Tag) override {
    auto *Record = dyn_cast<CXXRecordDecl>(Tag);
    if (!Record || Record->isCompleteDefinition())
      return;
    if (Record == Primary)
      ++PrimaryCompletions;
    else if (Record == Partial)
      ++PartialCompletions;
    else
      return;
    Record->setHasExternalLexicalStorage(false);
    Record->startDefinition();
    Record->completeDefinition();
  }
};

} // namespace

// An instantiation must be able to complete the pattern it is actually built
// from, whichever of the two that is, and in either order.
TEST(ExternalASTSourceTest, CompletesPatternInEitherOrder) {
  for (StringRef Code : {"A<int> a; A<int *> b;", "A<int *> b; A<int> a;"}) {
    auto Source = llvm::makeIntrusiveRefCnt<LazyTemplatePatterns>();
    ASSERT_TRUE(testExternalASTSource(Source, Code)) << Code;
    EXPECT_EQ(1u, Source->PrimaryCompletions) << Code;
    EXPECT_EQ(1u, Source->PartialCompletions) << Code;
  }
}
