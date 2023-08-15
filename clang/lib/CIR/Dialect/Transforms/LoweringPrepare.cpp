//===- LoweringPrepare.cpp - pareparation work for LLVM lowering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;

namespace {
struct LoweringPreparePass : public LoweringPrepareBase<LoweringPreparePass> {
  LoweringPreparePass() = default;
  void runOnOperation() override;

  ///
  /// AST related
  /// -----------

  clang::ASTContext *astCtx;
  void setASTContext(clang::ASTContext *c) { astCtx = c; }

  /// Tracks current module.
  ModuleOp theModule;
};
} // namespace


void LoweringPreparePass::runOnOperation() {
  assert(astCtx && "Missing ASTContext, please construct with the right ctor");
  auto* op = getOperation();
  if (isa<::mlir::ModuleOp>(op)) {
    theModule = cast<::mlir::ModuleOp>(op);
  }
}

std::unique_ptr<Pass> mlir::createLoweringPreparePass() {
  return std::make_unique<LoweringPreparePass>();
}

std::unique_ptr<Pass> mlir::createLoweringPreparePass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<LoweringPreparePass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
