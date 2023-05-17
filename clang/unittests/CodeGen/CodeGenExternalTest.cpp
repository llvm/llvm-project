//===- unittests/CodeGen/CodeGenExternalTest.cpp - test external CodeGen -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestCompiler.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/CodeGenABITypes.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/Sema.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

// Mocks up a language using Clang code generation as a library and
// tests some basic functionality there.
//   - CodeGen->GetAddrOfGlobal
//   - CodeGen::convertTypeForMemory
//   - CodeGen::getLLVMFieldNumber

static const bool DebugThisTest = false;

// forward declarations
struct MyASTConsumer;
static void test_codegen_fns(MyASTConsumer *my);
static bool test_codegen_fns_ran;

// This forwards the calls to the Clang CodeGenerator
// so that we can test CodeGen functions while it is open.
// It accumulates toplevel decls in HandleTopLevelDecl and
// calls test_codegen_fns() in HandleTranslationUnit
// before forwarding that function to the CodeGenerator.

struct MyASTConsumer : public ASTConsumer {
  std::unique_ptr<CodeGenerator> Builder;
  std::vector<Decl*> toplevel_decls;

  MyASTConsumer(std::unique_ptr<CodeGenerator> Builder_in)
    : ASTConsumer(), Builder(std::move(Builder_in))
  {
  }

  ~MyASTConsumer() { }

  void Initialize(ASTContext &Context) override;
  void HandleCXXStaticMemberVarInstantiation(VarDecl *VD) override;
  bool HandleTopLevelDecl(DeclGroupRef D) override;
  void HandleInlineFunctionDefinition(FunctionDecl *D) override;
  void HandleInterestingDecl(DeclGroupRef D) override;
  void HandleTranslationUnit(ASTContext &Ctx) override;
  void HandleTagDeclDefinition(TagDecl *D) override;
  void HandleTagDeclRequiredDefinition(const TagDecl *D) override;
  void HandleCXXImplicitFunctionInstantiation(FunctionDecl *D) override;
  void HandleTopLevelDeclInObjCContainer(DeclGroupRef D) override;
  void HandleImplicitImportDecl(ImportDecl *D) override;
  void CompleteTentativeDefinition(VarDecl *D) override;
  void AssignInheritanceModel(CXXRecordDecl *RD) override;
  void HandleVTable(CXXRecordDecl *RD) override;
  ASTMutationListener *GetASTMutationListener() override;
  ASTDeserializationListener *GetASTDeserializationListener() override;
  void PrintStats() override;
  bool shouldSkipFunctionBody(Decl *D) override;
};

void MyASTConsumer::Initialize(ASTContext &Context) {
  Builder->Initialize(Context);
}

bool MyASTConsumer::HandleTopLevelDecl(DeclGroupRef DG) {

  for (DeclGroupRef::iterator I = DG.begin(), E = DG.end(); I != E; ++I) {
    toplevel_decls.push_back(*I);
  }

  return Builder->HandleTopLevelDecl(DG);
}

void MyASTConsumer::HandleInlineFunctionDefinition(FunctionDecl *D) {
  Builder->HandleInlineFunctionDefinition(D);
}

void MyASTConsumer::HandleInterestingDecl(DeclGroupRef D) {
  Builder->HandleInterestingDecl(D);
}

void MyASTConsumer::HandleTranslationUnit(ASTContext &Context) {
  test_codegen_fns(this);
  // HandleTranslationUnit can close the module
  Builder->HandleTranslationUnit(Context);
}

void MyASTConsumer::HandleTagDeclDefinition(TagDecl *D) {
  Builder->HandleTagDeclDefinition(D);
}

void MyASTConsumer::HandleTagDeclRequiredDefinition(const TagDecl *D) {
  Builder->HandleTagDeclRequiredDefinition(D);
}

void MyASTConsumer::HandleCXXImplicitFunctionInstantiation(FunctionDecl *D) {
  Builder->HandleCXXImplicitFunctionInstantiation(D);
}

void MyASTConsumer::HandleTopLevelDeclInObjCContainer(DeclGroupRef D) {
  Builder->HandleTopLevelDeclInObjCContainer(D);
}

void MyASTConsumer::HandleImplicitImportDecl(ImportDecl *D) {
  Builder->HandleImplicitImportDecl(D);
}

void MyASTConsumer::CompleteTentativeDefinition(VarDecl *D) {
  Builder->CompleteTentativeDefinition(D);
}

void MyASTConsumer::AssignInheritanceModel(CXXRecordDecl *RD) {
  Builder->AssignInheritanceModel(RD);
}

void MyASTConsumer::HandleCXXStaticMemberVarInstantiation(VarDecl *VD) {
   Builder->HandleCXXStaticMemberVarInstantiation(VD);
}

void MyASTConsumer::HandleVTable(CXXRecordDecl *RD) {
   Builder->HandleVTable(RD);
 }

ASTMutationListener *MyASTConsumer::GetASTMutationListener() {
  return Builder->GetASTMutationListener();
}

ASTDeserializationListener *MyASTConsumer::GetASTDeserializationListener() {
  return Builder->GetASTDeserializationListener();
}

void MyASTConsumer::PrintStats() {
  Builder->PrintStats();
}

bool MyASTConsumer::shouldSkipFunctionBody(Decl *D) {
  return Builder->shouldSkipFunctionBody(D);
}

const char TestProgram[] =
    "struct mytest_struct { char x; short y; char p; long z; };\n"
    "int mytest_fn(int x) { return x; }\n";

// This function has the real test code here
static void test_codegen_fns(MyASTConsumer *my) {

  bool mytest_fn_ok = false;
  bool mytest_struct_ok = false;

  CodeGen::CodeGenModule &CGM = my->Builder->CGM();

  for (auto decl : my->toplevel_decls ) {
    if (FunctionDecl *fd = dyn_cast<FunctionDecl>(decl)) {
      if (fd->getName() == "mytest_fn") {
        Constant *c = my->Builder->GetAddrOfGlobal(GlobalDecl(fd), false);
        // Verify that we got a function.
        ASSERT_TRUE(c != NULL);
        if (DebugThisTest) {
          c->print(dbgs(), true);
          dbgs() << "\n";
        }
        mytest_fn_ok = true;
      }
    } else if(clang::RecordDecl *rd = dyn_cast<RecordDecl>(decl)) {
      if (rd->getName() == "mytest_struct") {
        RecordDecl *def = rd->getDefinition();
        ASSERT_TRUE(def != NULL);
        const clang::Type *clangTy = rd->getCanonicalDecl()->getTypeForDecl();
        ASSERT_TRUE(clangTy != NULL);
        QualType qType = clangTy->getCanonicalTypeInternal();

        // Check convertTypeForMemory
        llvm::Type *llvmTy = CodeGen::convertTypeForMemory(CGM, qType);
        ASSERT_TRUE(llvmTy != NULL);
        if (DebugThisTest) {
          llvmTy->print(dbgs(), true);
          dbgs() << "\n";
        }

        auto* structTy = dyn_cast<llvm::StructType>(llvmTy);
        ASSERT_TRUE(structTy != NULL);

        // Check getLLVMFieldNumber
        FieldDecl *xField = NULL;
        FieldDecl *yField = NULL;
        FieldDecl *zField = NULL;

        for (auto field : rd->fields()) {
          if (field->getName() == "x") xField = field;
          if (field->getName() == "y") yField = field;
          if (field->getName() == "z") zField = field;
        }

        ASSERT_TRUE(xField != NULL);
        ASSERT_TRUE(yField != NULL);
        ASSERT_TRUE(zField != NULL);

        unsigned x = CodeGen::getLLVMFieldNumber(CGM, rd, xField);
        unsigned y = CodeGen::getLLVMFieldNumber(CGM, rd, yField);
        unsigned z = CodeGen::getLLVMFieldNumber(CGM, rd, zField);

        ASSERT_NE(x, y);
        ASSERT_NE(y, z);

        llvm::Type* xTy = structTy->getTypeAtIndex(x);
        llvm::Type* yTy = structTy->getTypeAtIndex(y);
        llvm::Type* zTy = structTy->getTypeAtIndex(z);

        ASSERT_TRUE(xTy != NULL);
        ASSERT_TRUE(yTy != NULL);
        ASSERT_TRUE(zTy != NULL);

        if (DebugThisTest) {
          xTy->print(dbgs(), true);
          dbgs() << "\n";
          yTy->print(dbgs(), true);
          dbgs() << "\n";
          zTy->print(dbgs(), true);
          dbgs() << "\n";
        }

        ASSERT_GE(xTy->getPrimitiveSizeInBits(), 1u);
        ASSERT_GE(yTy->getPrimitiveSizeInBits(), 16u); // short is at least 16b
        ASSERT_GE(zTy->getPrimitiveSizeInBits(), 32u); // long is at least 32b

        mytest_struct_ok = true;
      }
    }
  }

  ASSERT_TRUE(mytest_fn_ok);
  ASSERT_TRUE(mytest_struct_ok);

  test_codegen_fns_ran = true;
}

TEST(CodeGenExternalTest, CodeGenExternalTest) {
  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  TestCompiler Compiler(LO);
  auto CustomASTConsumer
    = std::make_unique<MyASTConsumer>(std::move(Compiler.CG));

  Compiler.init(TestProgram, std::move(CustomASTConsumer));

  clang::ParseAST(Compiler.compiler.getSema(), false, false);

  ASSERT_TRUE(test_codegen_fns_ran);
}

} // end anonymous namespace
