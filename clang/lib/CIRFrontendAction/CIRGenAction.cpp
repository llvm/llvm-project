//===--- CIRGenAction.cpp - LLVM Code generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIRFrontendAction/CIRGenAction.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/CIRBuilder.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <memory>

using namespace cir;
using namespace clang;

namespace cir {
class CIRGenConsumer : public clang::ASTConsumer {

  virtual void anchor();
  ASTContext *astContext{nullptr};

  std::unique_ptr<CIRContext> gen;

public:
  CIRGenConsumer(std::unique_ptr<raw_pwrite_stream> os)
      : gen(std::make_unique<CIRContext>(std::move(os))) {}

  void Initialize(ASTContext &ctx) override {
    assert(!astContext && "initialized multiple times");

    astContext = &ctx;

    gen->Initialize(ctx);
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    PrettyStackTraceDecl CrashInfo(*D.begin(), SourceLocation(),
                                   astContext->getSourceManager(),
                                   "LLVM IR generation of declaration");
    gen->HandleTopLevelDecl(D);

    return true;
  }

  void HandleInlineFunctionDefinition(FunctionDecl *D) override {}

  void HandleInterestingDecl(DeclGroupRef D) override { HandleTopLevelDecl(D); }

  void HandleTranslationUnit(ASTContext &C) override {
    gen->HandleTranslationUnit(C);
    // TODO: Have context emit file here
  }

  void HandleTagDeclDefinition(TagDecl *D) override {}

  void HandleTagDeclRequiredDefinition(const TagDecl *D) override {}

  void CompleteTentativeDefinition(VarDecl *D) override {}

  void CompleteExternalDeclaration(DeclaratorDecl *D) override {}

  void AssignInheritanceModel(CXXRecordDecl *RD) override {}

  void HandleVTable(CXXRecordDecl *RD) override {}
};
} // namespace cir

void CIRGenConsumer::anchor() {}

CIRGenAction::CIRGenAction(OutputType act, mlir::MLIRContext *_MLIRContext)
    : MLIRContext(_MLIRContext ? _MLIRContext : new mlir::MLIRContext),
      OwnsVMContext(!_MLIRContext), action(act) {}

CIRGenAction::~CIRGenAction() {
  TheModule.reset();
  if (OwnsVMContext)
    delete MLIRContext;
}

void CIRGenAction::EndSourceFileAction() {}

static std::unique_ptr<raw_pwrite_stream>
getOutputStream(CompilerInstance &ci, StringRef inFile,
                CIRGenAction::OutputType action) {
  switch (action) {
  case CIRGenAction::OutputType::EmitAssembly:
    return ci.createDefaultOutputFile(false, inFile, "s");
  case CIRGenAction::OutputType::EmitCIR:
    return ci.createDefaultOutputFile(false, inFile, "cir");
  case CIRGenAction::OutputType::EmitLLVM:
    return ci.createDefaultOutputFile(true, inFile, "llvm");
  case CIRGenAction::OutputType::None:
    return nullptr;
  }

  llvm_unreachable("Invalid action!");
}

std::unique_ptr<ASTConsumer>
CIRGenAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  auto out = CI.takeOutputStream();
  if (!out)
    out = getOutputStream(CI, InFile, action);
  return std::make_unique<cir::CIRGenConsumer>(std::move(out));
}

std::unique_ptr<mlir::ModuleOp>
CIRGenAction::loadModule(llvm::MemoryBufferRef MBRef) {
  return {};
}

void CIRGenAction::ExecuteAction() { ASTFrontendAction::ExecuteAction(); }

void EmitAssemblyAction::anchor() {}
EmitAssemblyAction::EmitAssemblyAction(mlir::MLIRContext *_MLIRContext)
    : CIRGenAction(OutputType::EmitAssembly, _MLIRContext) {}

void EmitCIRAction::anchor() {}
EmitCIRAction::EmitCIRAction(mlir::MLIRContext *_MLIRContext)
    : CIRGenAction(OutputType::EmitCIR, _MLIRContext) {}

void EmitCIROnlyAction::anchor() {}
EmitCIROnlyAction::EmitCIROnlyAction(mlir::MLIRContext *_MLIRContext)
    : CIRGenAction(OutputType::None, _MLIRContext) {}
