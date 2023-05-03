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

#ifndef CLANG_CIRGENERATOR_H_
#define CLANG_CIRGENERATOR_H_

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/CodeGenOptions.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <memory>

namespace mlir {
class MLIRContext;
class ModuleOp;
class OwningModuleRef;
} // namespace mlir

namespace clang {
class ASTContext;
class DeclGroupRef;
class FunctionDecl;
} // namespace clang

namespace cir {
class CIRGenModule;
class CIRGenTypes;

class CIRGenerator : public clang::ASTConsumer {
  virtual void anchor();
  clang::DiagnosticsEngine &Diags;
  clang::ASTContext *astCtx;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
      fs; // Only used for debug info.

  const clang::CodeGenOptions codeGenOpts; // Intentionally copied in.

  unsigned HandlingTopLevelDecls;

  /// Use this when emitting decls to block re-entrant decl emission. It will
  /// emit all deferred decls on scope exit. Set EmitDeferred to false if decl
  /// emission must be deferred longer, like at the end of a tag definition.
  struct HandlingTopLevelDeclRAII {
    CIRGenerator &Self;
    bool EmitDeferred;
    HandlingTopLevelDeclRAII(CIRGenerator &Self, bool EmitDeferred = true)
        : Self{Self}, EmitDeferred{EmitDeferred} {
      ++Self.HandlingTopLevelDecls;
    }
    ~HandlingTopLevelDeclRAII() {
      unsigned Level = --Self.HandlingTopLevelDecls;
      if (Level == 0 && EmitDeferred)
        Self.buildDeferredDecls();
    }
  };

protected:
  std::unique_ptr<mlir::MLIRContext> mlirCtx;
  std::unique_ptr<CIRGenModule> CGM;

private:
  llvm::SmallVector<clang::FunctionDecl *, 8> DeferredInlineMemberFuncDefs;

public:
  CIRGenerator(clang::DiagnosticsEngine &diags,
               llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
               const clang::CodeGenOptions &CGO);
  ~CIRGenerator();
  void Initialize(clang::ASTContext &Context) override;
  bool EmitFunction(const clang::FunctionDecl *FD);

  bool HandleTopLevelDecl(clang::DeclGroupRef D) override;
  void HandleTranslationUnit(clang::ASTContext &Ctx) override;
  void HandleInlineFunctionDefinition(clang::FunctionDecl *D) override;
  void HandleTagDeclDefinition(clang::TagDecl *D) override;
  void HandleTagDeclRequiredDefinition(const clang::TagDecl *D) override;
  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *D) override;

  mlir::ModuleOp getModule();
  std::unique_ptr<mlir::MLIRContext> takeContext() {
    return std::move(mlirCtx);
  };

  bool verifyModule();

  void buildDeferredDecls();
  void buildDefaultMethods();
};

} // namespace cir

#endif // CLANG_CIRGENERATOR_H_
