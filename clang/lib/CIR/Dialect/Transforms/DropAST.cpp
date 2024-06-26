//===- DropAST.cpp - emit diagnostic checks for lifetime violations -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Passes.h"

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace cir;

namespace {
struct DropASTPass : public DropASTBase<DropASTPass> {
  DropASTPass() = default;
  void runOnOperation() override;
};
} // namespace

void DropASTPass::runOnOperation() {
  Operation *op = getOperation();
  // This needs to be updated with operations that start
  // carrying AST around.
  op->walk([&](Operation *op) {
    if (auto alloca = dyn_cast<AllocaOp>(op)) {
      alloca.removeAstAttr();
      auto ty = mlir::dyn_cast<mlir::cir::StructType>(alloca.getAllocaType());
      if (!ty)
        return;
      ty.dropAst();
      return;
    }

    if (auto funcOp = dyn_cast<FuncOp>(op))
      funcOp.removeAstAttr();
  });
}

std::unique_ptr<Pass> mlir::createDropASTPass() {
  return std::make_unique<DropASTPass>();
}
