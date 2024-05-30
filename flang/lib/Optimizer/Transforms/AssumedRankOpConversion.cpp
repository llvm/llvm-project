//===-- AssumedRankOpConversion.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"
#include "flang/Lower/BuiltinModules.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Support.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/TypeCode.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Runtime/support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_ASSUMEDRANKOPCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;

namespace {

static int getCFIAttribute(mlir::Type boxType) {
  if (fir::isAllocatableType(boxType))
    return CFI_attribute_allocatable;
  if (fir::isPointerType(boxType))
    return CFI_attribute_pointer;
  return CFI_attribute_other;
}

static Fortran::runtime::LowerBoundModifier
getLowerBoundModifier(fir::LowerBoundModifierAttribute modifier) {
  switch (modifier) {
  case fir::LowerBoundModifierAttribute::Preserve:
    return Fortran::runtime::LowerBoundModifier::Preserve;
  case fir::LowerBoundModifierAttribute::SetToOnes:
    return Fortran::runtime::LowerBoundModifier::SetToOnes;
  case fir::LowerBoundModifierAttribute::SetToZeroes:
    return Fortran::runtime::LowerBoundModifier::SetToZeroes;
  }
  llvm_unreachable("bad modifier code");
}

class ReboxAssumedRankConv
    : public mlir::OpRewritePattern<fir::ReboxAssumedRankOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  ReboxAssumedRankConv(mlir::MLIRContext *context,
                       mlir::SymbolTable *symbolTable, fir::KindMapping kindMap)
      : mlir::OpRewritePattern<fir::ReboxAssumedRankOp>(context),
        symbolTable{symbolTable}, kindMap{kindMap} {};

  mlir::LogicalResult
  matchAndRewrite(fir::ReboxAssumedRankOp rebox,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, kindMap, symbolTable};
    mlir::Location loc = rebox.getLoc();
    auto newBoxType = mlir::cast<fir::BaseBoxType>(rebox.getType());
    mlir::Type newMaxRankBoxType =
        newBoxType.getBoxTypeWithNewShape(Fortran::common::maxRank);
    // CopyAndUpdateDescriptor FIR interface requires loading
    // !fir.ref<fir.box> input which is expensive with assumed-rank. It could
    // be best to add an entry point that takes a non "const" from to cover
    // this case, but it would be good to indicate to LLVM that from does not
    // get modified.
    if (fir::isBoxAddress(rebox.getBox().getType()))
      TODO(loc, "fir.rebox_assumed_rank codegen with fir.ref<fir.box<>> input");
    mlir::Value tempDesc = builder.createTemporary(loc, newMaxRankBoxType);
    mlir::Value newDtype;
    mlir::Type newEleType = newBoxType.unwrapInnerType();
    auto oldBoxType = mlir::cast<fir::BaseBoxType>(
        fir::unwrapRefType(rebox.getBox().getType()));
    auto newDerivedType = mlir::dyn_cast<fir::RecordType>(newEleType);
    if (newDerivedType && (newEleType != oldBoxType.unwrapInnerType()) &&
        !fir::isPolymorphicType(newBoxType)) {
      newDtype = builder.create<fir::TypeDescOp>(
          loc, mlir::TypeAttr::get(newDerivedType));
    } else {
      newDtype = builder.createNullConstant(loc);
    }
    mlir::Value newAttribute = builder.createIntegerConstant(
        loc, builder.getIntegerType(8), getCFIAttribute(newBoxType));
    int lbsModifierCode =
        static_cast<int>(getLowerBoundModifier(rebox.getLbsModifier()));
    mlir::Value lowerBoundModifier = builder.createIntegerConstant(
        loc, builder.getIntegerType(32), lbsModifierCode);
    fir::runtime::genCopyAndUpdateDescriptor(builder, loc, tempDesc,
                                             rebox.getBox(), newDtype,
                                             newAttribute, lowerBoundModifier);

    mlir::Value descValue = builder.create<fir::LoadOp>(loc, tempDesc);
    mlir::Value castDesc = builder.createConvert(loc, newBoxType, descValue);
    rewriter.replaceOp(rebox, castDesc);
    return mlir::success();
  }

private:
  mlir::SymbolTable *symbolTable = nullptr;
  fir::KindMapping kindMap;
};

/// Convert FIR structured control flow ops to CFG ops.
class AssumedRankOpConversion
    : public fir::impl::AssumedRankOpConversionBase<AssumedRankOpConversion> {
public:
  void runOnOperation() override {
    auto *context = &getContext();
    mlir::ModuleOp mod = getOperation();
    mlir::SymbolTable symbolTable(mod);
    fir::KindMapping kindMap = fir::getKindMapping(mod);
    mlir::RewritePatternSet patterns(context);
    patterns.insert<ReboxAssumedRankConv>(context, &symbolTable, kindMap);
    mlir::GreedyRewriteConfig config;
    config.enableRegionSimplification = false;
    (void)applyPatternsAndFoldGreedily(mod, std::move(patterns), config);
  }
};
} // namespace
