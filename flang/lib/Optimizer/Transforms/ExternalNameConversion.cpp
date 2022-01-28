//===- ExternalNameConversion.cpp -- convert name with external convention ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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
    auto callee = op.callee();
    if (callee.hasValue()) {
      auto result = fir::NameUniquer::deconstruct(
          callee.getValue().getRootReference().getValue());
      if (fir::NameUniquer::isExternalFacingUniquedName(result))
        op.calleeAttr(
            SymbolRefAttr::get(op.getContext(), mangleExternalName(result)));
    }
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

struct MangleNameOnFuncOp : public mlir::OpRewritePattern<mlir::FuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::FuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    auto result = fir::NameUniquer::deconstruct(op.sym_name());
    if (fir::NameUniquer::isExternalFacingUniquedName(result)) {
      auto newName = mangleExternalName(result);
      op.sym_nameAttr(rewriter.getStringAttr(newName));
      SymbolTable::setSymbolName(op, newName);
    }
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
        op.symref().getRootReference().getValue());
    if (fir::NameUniquer::isExternalFacingUniquedName(result)) {
      auto newName = mangleExternalName(result);
      op.symrefAttr(mlir::SymbolRefAttr::get(op.getContext(), newName));
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
        op.symbol().getRootReference().getValue());
    if (fir::NameUniquer::isExternalFacingUniquedName(result)) {
      auto newName =
          SymbolRefAttr::get(op.getContext(), mangleExternalName(result));
      rewriter.replaceOpWithNewOp<fir::AddrOfOp>(op, op.resTy().getType(),
                                                 newName);
    }
    return success();
  }
};

struct MangleNameOnEmboxProcOp
    : public mlir::OpRewritePattern<fir::EmboxProcOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxProcOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    auto result = fir::NameUniquer::deconstruct(
        op.funcname().getRootReference().getValue());
    if (fir::NameUniquer::isExternalFacingUniquedName(result))
      op.funcnameAttr(
          SymbolRefAttr::get(op.getContext(), mangleExternalName(result)));
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

class ExternalNameConversionPass
    : public fir::ExternalNameConversionBase<ExternalNameConversionPass> {
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
                  MangleNameForCommonBlock, MangleNameOnAddrOfOp,
                  MangleNameOnEmboxProcOp>(context);

  ConversionTarget target(*context);
  target.addLegalDialect<fir::FIROpsDialect, LLVM::LLVMDialect,
                         acc::OpenACCDialect, omp::OpenMPDialect>();

  target.addDynamicallyLegalOp<fir::CallOp>([](fir::CallOp op) {
    if (op.callee().hasValue())
      return !fir::NameUniquer::needExternalNameMangling(
          op.callee().getValue().getRootReference().getValue());
    return true;
  });

  target.addDynamicallyLegalOp<mlir::FuncOp>([](mlir::FuncOp op) {
    return !fir::NameUniquer::needExternalNameMangling(op.sym_name());
  });

  target.addDynamicallyLegalOp<fir::GlobalOp>([](fir::GlobalOp op) {
    return !fir::NameUniquer::needExternalNameMangling(
        op.symref().getRootReference().getValue());
  });

  target.addDynamicallyLegalOp<fir::AddrOfOp>([](fir::AddrOfOp op) {
    return !fir::NameUniquer::needExternalNameMangling(
        op.symbol().getRootReference().getValue());
  });

  target.addDynamicallyLegalOp<fir::EmboxProcOp>([](fir::EmboxProcOp op) {
    return !fir::NameUniquer::needExternalNameMangling(
        op.funcname().getRootReference().getValue());
  });

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> fir::createExternalNameConversionPass() {
  return std::make_unique<ExternalNameConversionPass>();
}
