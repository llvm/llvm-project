//===- CIRBuilder.h - CIR Generation from Clang AST -----------------------===//
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

#ifndef CLANG_CIRBUILDER_H_
#define CLANG_CIRBUILDER_H_

#include <memory>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace clang {
class ASTContext;
class FunctionDecl;
} // namespace clang

namespace cir {
class CIRBuildImpl;
}

namespace cir {

class CIRContext {
public:
  ~CIRContext();
  CIRContext(clang::ASTContext &AC);
  void Init();
  bool EmitFunction(const clang::FunctionDecl *FD);

private:
  std::unique_ptr<mlir::MLIRContext> mlirCtx;
  std::unique_ptr<CIRBuildImpl> builder;
  clang::ASTContext &astCtx;
};

} // namespace cir

#endif // CLANG_CIRBUILDER_H_