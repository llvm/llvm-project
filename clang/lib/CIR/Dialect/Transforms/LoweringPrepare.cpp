//===- LoweringPrepare.cpp - pareparation work for LLVM lowering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"

#include <memory>

using namespace mlir;
using namespace cir;

namespace {
struct LoweringPreparePass : public LoweringPrepareBase<LoweringPreparePass> {
  LoweringPreparePass() = default;
  void runOnOperation() override;

  void runOnOp(Operation *op);
  void lowerUnaryOp(UnaryOp op);
};

} // namespace

void LoweringPreparePass::lowerUnaryOp(UnaryOp op) {
  mlir::Type ty = op.getType();
  if (!mlir::isa<cir::ComplexType>(ty))
    return;

  mlir::Location loc = op.getLoc();
  cir::UnaryOpKind opKind = op.getKind();

  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);

  mlir::Value operand = op.getInput();
  mlir::Value operandReal = builder.createComplexReal(loc, operand);
  mlir::Value operandImag = builder.createComplexImag(loc, operand);

  mlir::Value resultReal;
  mlir::Value resultImag;

  switch (opKind) {
  case cir::UnaryOpKind::Inc:
  case cir::UnaryOpKind::Dec:
    llvm_unreachable("Complex unary Inc/Dec NYI");
    break;

  case cir::UnaryOpKind::Plus:
  case cir::UnaryOpKind::Minus:
    llvm_unreachable("Complex unary Plus/Minus NYI");
    break;

  case cir::UnaryOpKind::Not:
    resultReal = operandReal;
    resultImag =
        builder.createUnaryOp(loc, cir::UnaryOpKind::Minus, operandImag);
    break;
  }

  auto result = builder.createComplexCreate(loc, resultReal, resultImag);
  op.replaceAllUsesWith(result);
  op.erase();
}

void LoweringPreparePass::runOnOp(Operation *op) {
  if (auto unary = dyn_cast<UnaryOp>(op)) {
    lowerUnaryOp(unary);
  }
}

void LoweringPreparePass::runOnOperation() {
  mlir::Operation *op = getOperation();

  llvm::SmallVector<Operation *> opsToTransform;

  op->walk([&](Operation *op) {
    if (isa<UnaryOp>(op))
      opsToTransform.push_back(op);
  });

  for (auto *o : opsToTransform)
    runOnOp(o);
}

std::unique_ptr<Pass> mlir::createLoweringPreparePass() {
  return std::make_unique<LoweringPreparePass>();
}
