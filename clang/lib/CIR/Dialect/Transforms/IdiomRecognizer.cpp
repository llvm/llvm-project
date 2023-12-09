//===- IdiomRecognizer.cpp - pareparation work for LLVM lowering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

using cir::CIRBaseBuilderTy;
using namespace mlir;
using namespace mlir::cir;

namespace {

struct IdiomRecognizerPass : public IdiomRecognizerBase<IdiomRecognizerPass> {
  IdiomRecognizerPass() = default;
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

void IdiomRecognizerPass::runOnOperation() {
  assert(astCtx && "Missing ASTContext, please construct with the right ctor");
  auto *op = getOperation();
  if (isa<::mlir::ModuleOp>(op))
    theModule = cast<::mlir::ModuleOp>(op);
}

std::unique_ptr<Pass> mlir::createIdiomRecognizerPass() {
  return std::make_unique<IdiomRecognizerPass>();
}

std::unique_ptr<Pass>
mlir::createIdiomRecognizerPass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<IdiomRecognizerPass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
