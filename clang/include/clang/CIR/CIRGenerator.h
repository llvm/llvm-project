//===- CIRGenerator.h - CIR Generation from Clang AST ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform CIR generation from Clang
// AST
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_CIRGENERATOR_H
#define LLVM_CLANG_CIR_CIRGENERATOR_H

#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/CodeGenOptions.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <memory>

namespace clang {
class DeclGroupRef;
class DiagnosticsEngine;
namespace CIRGen {
class CIRGenModule;
} // namespace CIRGen
} // namespace clang

namespace mlir {
class MLIRContext;
} // namespace mlir
namespace cir {
class CIRGenerator : public clang::ASTConsumer {
  virtual void anchor();
  clang::DiagnosticsEngine &diags;
  clang::ASTContext *astContext;
  // Only used for debug info.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs;

  const clang::CodeGenOptions &codeGenOpts;

  unsigned handlingTopLevelDecls;

  /// Use this when emitting decls to block re-entrant decl emission. It will
  /// emit all deferred decls on scope exit. Set EmitDeferred to false if decl
  /// emission must be deferred longer, like at the end of a tag definition.
  struct HandlingTopLevelDeclRAII {
    CIRGenerator &self;
    bool emitDeferred;
    HandlingTopLevelDeclRAII(CIRGenerator &self, bool emitDeferred = true)
        : self{self}, emitDeferred{emitDeferred} {
      ++self.handlingTopLevelDecls;
    }
    ~HandlingTopLevelDeclRAII() {
      unsigned Level = --self.handlingTopLevelDecls;
      if (Level == 0 && emitDeferred)
        self.emitDeferredDecls();
    }
  };

protected:
  std::unique_ptr<mlir::MLIRContext> mlirContext;
  std::unique_ptr<clang::CIRGen::CIRGenModule> cgm;

private:
  llvm::SmallVector<clang::FunctionDecl *, 8> deferredInlineMemberFuncDefs;

public:
  CIRGenerator(clang::DiagnosticsEngine &diags,
               llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
               const clang::CodeGenOptions &cgo);
  ~CIRGenerator() override;
  void Initialize(clang::ASTContext &astContext) override;
  bool HandleTopLevelDecl(clang::DeclGroupRef group) override;
  void HandleTranslationUnit(clang::ASTContext &astContext) override;
  void HandleInlineFunctionDefinition(clang::FunctionDecl *d) override;
  void HandleTagDeclDefinition(clang::TagDecl *d) override;
  void HandleTagDeclRequiredDefinition(const clang::TagDecl *D) override;
  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *D) override;
  void CompleteTentativeDefinition(clang::VarDecl *d) override;
  void HandleVTable(clang::CXXRecordDecl *rd) override;

  mlir::ModuleOp getModule() const;
  mlir::MLIRContext &getMLIRContext() { return *mlirContext; };
  const mlir::MLIRContext &getMLIRContext() const { return *mlirContext; };

  bool verifyModule() const;

  void emitDeferredDecls();
};

} // namespace cir

#endif // LLVM_CLANG_CIR_CIRGENERATOR_H
