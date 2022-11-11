//===- ExternalNameConversion.cpp -- convert name with external convention ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
#define GEN_PASS_DEF_EXTERNALNAMECONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Mangle the name with gfortran convention.
std::string
mangleExternalName(const std::pair<fir::NameUniquer::NameKind,
                                   fir::NameUniquer::DeconstructedName>
                       result) {
  if (result.first == fir::NameUniquer::NameKind::COMMON &&
      result.second.name.empty())
    return "__BLNK__";
  return result.second.name + "_";
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

namespace {

class MangleNameOnCallOp : public mlir::OpRewritePattern<fir::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    auto callee = op.getCallee();
    if (callee) {
      auto result =
          fir::NameUniquer::deconstruct(callee->getRootReference().getValue());
      if (fir::NameUniquer::isExternalFacingUniquedName(result))
        op.setCalleeAttr(
            SymbolRefAttr::get(op.getContext(), mangleExternalName(result)));
    }
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

struct MangleNameOnFuncOp : public mlir::OpRewritePattern<mlir::func::FuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    auto result = fir::NameUniquer::deconstruct(op.getSymName());
    if (fir::NameUniquer::isExternalFacingUniquedName(result))
      op.setSymNameAttr(rewriter.getStringAttr(mangleExternalName(result)));
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

struct MangleNameForCommonBlock : public mlir::OpRewritePattern<fir::GlobalOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::GlobalOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    auto result = fir::NameUniquer::deconstruct(
        op.getSymref().getRootReference().getValue());
    if (fir::NameUniquer::isExternalFacingUniquedName(result)) {
      auto newName = mangleExternalName(result);
      op.setSymrefAttr(mlir::SymbolRefAttr::get(op.getContext(), newName));
      SymbolTable::setSymbolName(op, newName);
    }
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

struct MangleNameOnAddrOfOp : public mlir::OpRewritePattern<fir::AddrOfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::AddrOfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto result = fir::NameUniquer::deconstruct(
        op.getSymbol().getRootReference().getValue());
    if (fir::NameUniquer::isExternalFacingUniquedName(result)) {
      auto newName =
          SymbolRefAttr::get(op.getContext(), mangleExternalName(result));
      rewriter.replaceOpWithNewOp<fir::AddrOfOp>(op, op.getResTy().getType(),
                                                 newName);
    }
    return success();
  }
};

class ExternalNameConversionPass
    : public fir::impl::ExternalNameConversionBase<ExternalNameConversionPass> {
public:
  mlir::ModuleOp getModule() { return getOperation(); }
  void runOnOperation() override;
};
} // namespace

void ExternalNameConversionPass::runOnOperation() {
  auto op = getOperation();
  auto *context = &getContext();

  mlir::RewritePatternSet patterns(context);
  patterns.insert<MangleNameOnCallOp, MangleNameOnCallOp, MangleNameOnFuncOp,
                  MangleNameForCommonBlock, MangleNameOnAddrOfOp>(context);

  ConversionTarget target(*context);
  target.addLegalDialect<fir::FIROpsDialect, LLVM::LLVMDialect,
                         acc::OpenACCDialect, omp::OpenMPDialect>();

  target.addDynamicallyLegalOp<fir::CallOp>([](fir::CallOp op) {
    if (op.getCallee())
      return !fir::NameUniquer::needExternalNameMangling(
          op.getCallee()->getRootReference().getValue());
    return true;
  });

  target.addDynamicallyLegalOp<mlir::func::FuncOp>([](mlir::func::FuncOp op) {
    return !fir::NameUniquer::needExternalNameMangling(op.getSymName());
  });

  target.addDynamicallyLegalOp<fir::GlobalOp>([](fir::GlobalOp op) {
    return !fir::NameUniquer::needExternalNameMangling(
        op.getSymref().getRootReference().getValue());
  });

  target.addDynamicallyLegalOp<fir::AddrOfOp>([](fir::AddrOfOp op) {
    return !fir::NameUniquer::needExternalNameMangling(
        op.getSymbol().getRootReference().getValue());
  });

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> fir::createExternalNameConversionPass() {
  return std::make_unique<ExternalNameConversionPass>();
}
