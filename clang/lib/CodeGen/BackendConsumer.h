//===--- BackendConsumer.h - LLVM BackendConsumer Header File -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_BACKENDCONSUMER_H
#define LLVM_CLANG_LIB_CODEGEN_BACKENDCONSUMER_H

#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleLinker.h"
#include "clang/CodeGenUtils/BackendDiagnosticHandler.h"

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/Timer.h"

namespace llvm {
  class DiagnosticInfoDontCall;
}

namespace clang {
class ASTContext;
class CodeGenAction;
class CoverageSourceInfo;

class BackendConsumer : public ASTConsumer {
  virtual void anchor();
  CompilerInstance &CI;
  DiagnosticsEngine &Diags;
  const CodeGenOptions &CodeGenOpts;
  const TargetOptions &TargetOpts;
  const LangOptions &LangOpts;
  std::unique_ptr<raw_pwrite_stream> AsmOutStream;
  ASTContext *Context = nullptr;
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;

  llvm::Timer LLVMIRGeneration;
  unsigned LLVMIRGenerationRefCount = 0;

  bool TimerIsEnabled = false;

  BackendAction Action;

  std::unique_ptr<CodeGenerator> Gen;

  SmallVector<LinkModule, 4> LinkModules;

  // Translates LLVM backend diagnostics into clang diagnostics; shared with
  // CIR so that both LLVM-emitting pipelines report backend diagnostics
  // through the same mechanism.
  BackendDiagnosticConsumer DiagConsumer;

public:
  BackendConsumer(CompilerInstance &CI, BackendAction Action,
                  IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                  llvm::LLVMContext &C, SmallVector<LinkModule, 4> LinkModules,
                  StringRef InFile, std::unique_ptr<raw_pwrite_stream> OS,
                  CoverageSourceInfo *CoverageInfo,
                  llvm::Module *CurLinkModule = nullptr);

  llvm::Module *getModule() const;
  std::unique_ptr<llvm::Module> takeModule();

  CodeGenerator *getCodeGenerator();

  void HandleCXXStaticMemberVarInstantiation(VarDecl *VD) override;
  void Initialize(ASTContext &Ctx) override;
  bool HandleTopLevelDecl(DeclGroupRef D) override;
  void HandleInlineFunctionDefinition(FunctionDecl *D) override;
  void HandleInterestingDecl(DeclGroupRef D) override;
  void HandleTranslationUnit(ASTContext &C) override;
  void HandleTagDeclDefinition(TagDecl *D) override;
  void HandleTagDeclRequiredDefinition(const TagDecl *D) override;
  void CompleteTentativeDefinition(VarDecl *D) override;
  void CompleteExternalDeclaration(DeclaratorDecl *D) override;
  void AssignInheritanceModel(CXXRecordDecl *RD) override;
  void HandleVTable(CXXRecordDecl *RD) override;

  // Links each entry in LinkModules into our module.  Returns true on error.
  bool LinkInModules(llvm::Module *M);

  /// Create an llvm::DiagnosticHandler that routes LLVM backend diagnostics
  /// through this consumer's clang diagnostics.
  std::unique_ptr<llvm::DiagnosticHandler> createDiagnosticHandler();
};

} // namespace clang
#endif
