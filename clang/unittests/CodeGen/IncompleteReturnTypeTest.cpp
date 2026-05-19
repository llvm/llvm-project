//===- unittests/CodeGen/IncompleteReturnTypeTest.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Regression test for de82b4790943: calling hasTrivialCopyConstructor() on an
// incomplete CXXRecordDecl in withReturnValueSlot (CGExprAgg.cpp) must not
// assert. This test synthesises a CallExpr whose return type is an incomplete
// C++ class, bypassing Sema's completeness check, and feeds it to CodeGen.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Parse/ParseAST.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

// An ASTConsumer that intercepts HandleTranslationUnit to inject a synthetic
// function (with a call returning an incomplete type) into CodeGen.
struct InjectIncompleteReturnConsumer : public ASTConsumer {
  std::unique_ptr<CodeGenerator> Builder;

  InjectIncompleteReturnConsumer(std::unique_ptr<CodeGenerator> B)
      : Builder(std::move(B)) {}

  void Initialize(ASTContext &Context) override {
    Builder->Initialize(Context);
  }

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    return Builder->HandleTopLevelDecl(DG);
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    // Create a forward-declared C++ class (no definition).
    IdentifierInfo &ClassId = Ctx.Idents.get("IncompleteClass");
    auto *RD = CXXRecordDecl::Create(
        Ctx, TagTypeKind::Class, Ctx.getTranslationUnitDecl(), SourceLocation(),
        SourceLocation(), &ClassId, /*PrevDecl=*/nullptr);
    Ctx.getTranslationUnitDecl()->addDecl(RD);
    QualType ClassTy = Ctx.getCanonicalTagType(RD);

    // Create a function declaration: IncompleteClass make();
    IdentifierInfo &FnId = Ctx.Idents.get("__test_make_incomplete");
    QualType FnTy =
        Ctx.getFunctionType(ClassTy, {}, FunctionProtoType::ExtProtoInfo());
    auto *MakeFD = FunctionDecl::Create(
        Ctx, Ctx.getTranslationUnitDecl(), SourceLocation(), SourceLocation(),
        DeclarationName(&FnId), FnTy, Ctx.getTrivialTypeSourceInfo(FnTy),
        SC_Extern);
    Ctx.getTranslationUnitDecl()->addDecl(MakeFD);

    // Create a DeclRefExpr referencing 'make'.
    QualType FnPtrTy = Ctx.getPointerType(FnTy);
    auto *DRE = DeclRefExpr::Create(
        Ctx, NestedNameSpecifierLoc(), SourceLocation(), MakeFD,
        /*RefersToEnclosingVariableOrCapture=*/false, SourceLocation(), FnTy,
        VK_LValue, /*FoundD=*/nullptr, /*TemplateArgs=*/nullptr);

    // Wrap in implicit FunctionToPointerDecay cast.
    auto *FnCast = ImplicitCastExpr::Create(
        Ctx, FnPtrTy, CK_FunctionToPointerDecay, DRE, /*BasePath=*/nullptr,
        VK_PRValue, FPOptionsOverride());

    // Create the CallExpr: make()
    auto *Call = CallExpr::Create(Ctx, FnCast, /*Args=*/{}, ClassTy, VK_PRValue,
                                  SourceLocation(), FPOptionsOverride());

    // Wrap the call in a CompoundStmt.
    Stmt *Stmts[] = {Call};
    auto *Body = CompoundStmt::Create(Ctx, Stmts, FPOptionsOverride(),
                                      SourceLocation(), SourceLocation());

    // Create wrapper function: void __test_caller() { make(); }
    IdentifierInfo &CallerId = Ctx.Idents.get("__test_caller");
    QualType VoidFnTy =
        Ctx.getFunctionType(Ctx.VoidTy, {}, FunctionProtoType::ExtProtoInfo());
    auto *CallerFD = FunctionDecl::Create(
        Ctx, Ctx.getTranslationUnitDecl(), SourceLocation(), SourceLocation(),
        DeclarationName(&CallerId), VoidFnTy,
        Ctx.getTrivialTypeSourceInfo(VoidFnTy), SC_None);
    CallerFD->setBody(Body);
    Ctx.getTranslationUnitDecl()->addDecl(CallerFD);

    // Feed the caller to CodeGen. This will trigger withReturnValueSlot
    // on an incomplete CXXRecordDecl.
    DeclGroupRef DG(CallerFD);
    Builder->HandleTopLevelDecl(DG);

    Builder->HandleTranslationUnit(Ctx);
  }
};

static void runIncompleteReturnTypeCodeGen() {
  LLVMContext Ctx;
  CompilerInstance CI;

  CI.getLangOpts().CPlusPlus = 1;
  CI.getLangOpts().CPlusPlus11 = 1;
  CI.setVirtualFileSystem(vfs::getRealFileSystem());
  CI.createDiagnostics();

  CI.getTargetOpts().Triple = "spir64-unknown-unknown";
  CI.setTarget(
      TargetInfo::CreateTargetInfo(CI.getDiagnostics(), CI.getTargetOpts()));

  CI.createFileManager();
  CI.createSourceManager();
  CI.createPreprocessor(TU_Prefix);
  CI.createASTContext();

  auto CG = CreateLLVMCodeGen(CI, "test-module", Ctx);
  auto Consumer =
      std::make_unique<InjectIncompleteReturnConsumer>(std::move(CG));
  CI.setASTConsumer(std::move(Consumer));
  CI.createSema(TU_Prefix, nullptr);

  SourceManager &SM = CI.getSourceManager();
  SM.setMainFileID(
      SM.createFileID(MemoryBuffer::getMemBuffer(""), SrcMgr::C_User));

  clang::ParseAST(CI.getSema(), false, false);
}

// This test verifies that withReturnValueSlot (CGExprAgg.cpp) does not assert
// "queried property of class with no definition" when encountering a call that
// returns an incomplete C++ class type.
//
// After the fix, CodeGen may still crash later (in ABI classification for the
// incomplete type), but the specific assertion in withReturnValueSlot must not
// fire. We use ASSERT_DEATH to verify the crash message is NOT the one from
// CXXRecordDecl::data().
TEST(IncompleteReturnTypeTest, NoAssertOnIncompleteReturnType) {
  ASSERT_DEATH(runIncompleteReturnTypeCodeGen(),
               "Cannot get layout of forward declarations");
}

} // end anonymous namespace
