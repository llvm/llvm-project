//===- ExternalNameConversion.cpp -- convert name with external convention ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"
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
                       result,
                   bool appendUnderscore) {
  if (result.first == fir::NameUniquer::NameKind::COMMON &&
      result.second.name.empty())
    return Fortran::common::blankCommonObjectName;

  if (appendUnderscore)
    return result.second.name + "_";

  return result.second.name;
}

/// Update the early outlining parent name
void updateEarlyOutliningParentName(mlir::func::FuncOp funcOp,
                                    bool appendUnderscore) {
  if (auto earlyOutlineOp = llvm::dyn_cast<mlir::omp::EarlyOutliningInterface>(
          funcOp.getOperation())) {
    auto oldName = earlyOutlineOp.getParentName();
    if (oldName != "") {
      auto dName = fir::NameUniquer::deconstruct(oldName);
      std::string newName = mangleExternalName(dName, appendUnderscore);
      earlyOutlineOp.setParentName(newName);
    }
  }
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

namespace {

struct MangleNameOnFuncOp : public mlir::OpRewritePattern<mlir::func::FuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  MangleNameOnFuncOp(mlir::MLIRContext *ctx, bool appendUnderscore)
      : mlir::OpRewritePattern<mlir::func::FuncOp>(ctx),
        appendUnderscore(appendUnderscore) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::LogicalResult ret = success();
    rewriter.startRootUpdate(op);
    auto result = fir::NameUniquer::deconstruct(op.getSymName());
    if (fir::NameUniquer::isExternalFacingUniquedName(result)) {
      auto newSymbol =
          rewriter.getStringAttr(mangleExternalName(result, appendUnderscore));

      // Try to update all SymbolRef's in the module that match the current op
      if (mlir::ModuleOp mod = op->getParentOfType<mlir::ModuleOp>())
        ret = op.replaceAllSymbolUses(newSymbol, mod);

      op.setSymNameAttr(newSymbol);
      mlir::SymbolTable::setSymbolName(op, newSymbol);
    }

    updateEarlyOutliningParentName(op, appendUnderscore);
    rewriter.finalizeRootUpdate(op);
    return ret;
  }

private:
  bool appendUnderscore;
};

struct MangleNameForCommonBlock : public mlir::OpRewritePattern<fir::GlobalOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  MangleNameForCommonBlock(mlir::MLIRContext *ctx, bool appendUnderscore)
      : mlir::OpRewritePattern<fir::GlobalOp>(ctx),
        appendUnderscore(appendUnderscore) {}

  mlir::LogicalResult
  matchAndRewrite(fir::GlobalOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    auto result = fir::NameUniquer::deconstruct(
        op.getSymref().getRootReference().getValue());
    if (fir::NameUniquer::isExternalFacingUniquedName(result)) {
      auto newName = mangleExternalName(result, appendUnderscore);
      op.setSymrefAttr(mlir::SymbolRefAttr::get(op.getContext(), newName));
      SymbolTable::setSymbolName(op, newName);
    }
    rewriter.finalizeRootUpdate(op);
    return success();
  }

private:
  bool appendUnderscore;
};

struct MangleNameOnAddrOfOp : public mlir::OpRewritePattern<fir::AddrOfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  MangleNameOnAddrOfOp(mlir::MLIRContext *ctx, bool appendUnderscore)
      : mlir::OpRewritePattern<fir::AddrOfOp>(ctx),
        appendUnderscore(appendUnderscore) {}

  mlir::LogicalResult
  matchAndRewrite(fir::AddrOfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto result = fir::NameUniquer::deconstruct(
        op.getSymbol().getRootReference().getValue());
    if (fir::NameUniquer::isExternalFacingUniquedName(result)) {
      auto newName = SymbolRefAttr::get(
          op.getContext(), mangleExternalName(result, appendUnderscore));
      rewriter.replaceOpWithNewOp<fir::AddrOfOp>(op, op.getResTy().getType(),
                                                 newName);
    }
    return success();
  }

private:
  bool appendUnderscore;
};

class ExternalNameConversionPass
    : public fir::impl::ExternalNameConversionBase<ExternalNameConversionPass> {
public:
  ExternalNameConversionPass(bool appendUnderscoring)
      : appendUnderscores(appendUnderscoring) {}

  ExternalNameConversionPass() { usePassOpt = true; }

  mlir::ModuleOp getModule() { return getOperation(); }
  void runOnOperation() override;

private:
  bool appendUnderscores;
  bool usePassOpt = false;
};
} // namespace

void ExternalNameConversionPass::runOnOperation() {
  auto op = getOperation();
  auto *context = &getContext();

  appendUnderscores = (usePassOpt) ? appendUnderscoreOpt : appendUnderscores;

  mlir::RewritePatternSet patterns(context);
  patterns.insert<MangleNameOnFuncOp, MangleNameForCommonBlock,
                  MangleNameOnAddrOfOp>(context, appendUnderscores);

  ConversionTarget target(*context);
  target.addLegalDialect<fir::FIROpsDialect, LLVM::LLVMDialect,
                         acc::OpenACCDialect, omp::OpenMPDialect>();

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

std::unique_ptr<mlir::Pass>
fir::createExternalNameConversionPass(bool appendUnderscoring) {
  return std::make_unique<ExternalNameConversionPass>(appendUnderscoring);
}
