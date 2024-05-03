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
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/Diagnostic.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <memory>

namespace mlir {
class MLIRContext;
} // namespace mlir
namespace cir {
class CIRGenModule;

class CIRGenerator : public clang::ASTConsumer {
  virtual void anchor();
  clang::DiagnosticsEngine &Diags;
  clang::ASTContext *astCtx;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
      fs; // Only used for debug info.

  const clang::CodeGenOptions codeGenOpts; // Intentionally copied in.

  [[maybe_unused]] unsigned HandlingTopLevelDecls;

protected:
  std::unique_ptr<mlir::MLIRContext> mlirCtx;
  std::unique_ptr<CIRGenModule> CGM;

public:
  CIRGenerator(clang::DiagnosticsEngine &diags,
               llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
               const clang::CodeGenOptions &CGO);
  ~CIRGenerator();
  void Initialize(clang::ASTContext &Context) override;
  bool HandleTopLevelDecl(clang::DeclGroupRef D) override;
};

} // namespace cir

#endif // CLANG_CIRGENERATOR_H_
