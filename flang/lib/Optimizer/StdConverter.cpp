//===-- StdConverter.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

// This module performs the conversion of some FIR operations.
// Convert some FIR types to standard dialect?

static llvm::cl::opt<bool> disableFirToStd(
    "disable-fir-to-std",
    llvm::cl::desc("disable conversion of fir.select_type and affine dialect "
                   "to the standard dialect pass"),
    llvm::cl::init(false), llvm::cl::Hidden);

namespace fir {
namespace {

using SmallVecResult = llvm::SmallVector<mlir::Value, 4>;
using OperandTy = llvm::ArrayRef<mlir::Value>;
using AttributeTy = llvm::ArrayRef<mlir::NamedAttribute>;

/// FIR to standard type converter
/// This converts a subset of FIR types to standard types
class FIRToStdTypeConverter : public mlir::TypeConverter {
public:
  using TypeConverter::TypeConverter;

  explicit FIRToStdTypeConverter(KindMapping &map) : kindMap{map} {
    addConversion([&](CplxType type) {
      return mlir::ComplexType::get(toFloatType(type.getFKind()));
    });
    addConversion([&](RealType type) { return toFloatType(type.getFKind()); });
    addConversion([&](IntType type) { return toIntegerType(type.getFKind()); });
  }

private:
  mlir::Type toFloatType(KindTy kind) {
    auto *ctx = kindMap.getContext();
    switch (kindMap.getRealTypeID(kind)) {
    case llvm::Type::TypeID::HalfTyID:
      return mlir::FloatType::getF16(ctx);
#if 0
    // TODO: there is no BF16 type in LLVM yet, so add this when one becomes
    // available
    case llvm::Type::TypeID:: FIXME TyID:
      return mlir::FloatType::getBF16(ctx);
#endif
    case llvm::Type::TypeID::FloatTyID:
      return mlir::FloatType::getF32(ctx);
    case llvm::Type::TypeID::DoubleTyID:
      return mlir::FloatType::getF64(ctx);
    case llvm::Type::TypeID::X86_FP80TyID: // MLIR does not support yet
      [[fallthrough]];
    case llvm::Type::TypeID::FP128TyID: // MLIR does not support yet
      [[fallthrough]];
    default:
      return RealType::get(ctx, kind);
    }
  }

  mlir::Type toIntegerType(KindTy kind) {
    return mlir::IntegerType::get(kindMap.getIntegerBitsize(kind),
                                  kindMap.getContext());
  }

  // clang++ erroneously complains this variable is unused (see CMakeLists.txt)
  KindMapping &kindMap;
};

/// FIR conversion pattern template
template <typename FromOp>
class FIROpConversion : public mlir::ConversionPattern {
public:
  explicit FIROpConversion(
      mlir::MLIRContext *ctx /*, FIRToStdTypeConverter &lowering*/)
      : ConversionPattern(FromOp::getOperationName(), 1,
                          ctx) /*, lowering(lowering)*/
  {}

  static Block *createBlock(mlir::ConversionPatternRewriter &rewriter,
                            Block *insertBefore) {
    assert(insertBefore && "expected valid insertion block");
    return rewriter.createBlock(insertBefore->getParent(),
                                mlir::Region::iterator(insertBefore));
  }

protected:
  // mlir::Type convertType(mlir::Type ty) const { return
  // lowering.convertType(ty); }

  // FIRToStdTypeConverter &lowering;
};

/// SelectTypeOp converted to an if-then-else chain
///
/// This lowers the test conditions to calls into the runtime
struct SelectTypeOpConversion : public FIROpConversion<SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto selectType = mlir::cast<SelectTypeOp>(op);
    auto conds = selectType.getNumConditions();
    auto attrName = SelectTypeOp::getCasesAttr();
    auto caseAttr = selectType.getAttrOfType<mlir::ArrayAttr>(attrName);
    auto cases = caseAttr.getValue();
    // Selector must be of type !fir.box<T>
    auto selector = selectType.getSelector(operands);
    auto loc = selectType.getLoc();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    for (decltype(conds) t = 0; t != conds; ++t) {
      auto *dest = selectType.getSuccessor(t);
      auto destOps = selectType.getSuccessorOperands(operands, t);
      auto &attr = cases[t];
      if (auto a = attr.dyn_cast<ExactTypeAttr>()) {
        genTypeLadderStep(loc, /*exactTest=*/true, selector, a.getType(), dest,
                          destOps, mod, rewriter);
        continue;
      }
      if (auto a = attr.dyn_cast<SubclassAttr>()) {
        genTypeLadderStep(loc, /*exactTest=*/false, selector, a.getType(), dest,
                          destOps, mod, rewriter);
        continue;
      }
      assert(attr.isa<mlir::UnitAttr>());
      assert((t + 1 == conds) && "unit must be last");
      rewriter.replaceOpWithNewOp<mlir::BranchOp>(
          selectType, dest, mlir::ValueRange{destOps.getValue()});
    }
    return success();
  }

  static void genTypeLadderStep(mlir::Location loc, bool exactTest,
                                mlir::Value selector, mlir::Type ty,
                                mlir::Block *dest,
                                llvm::Optional<OperandTy> destOps,
                                mlir::ModuleOp module,
                                mlir::ConversionPatternRewriter &rewriter) {
    mlir::Type tydesc = TypeDescType::get(ty);
    auto tyattr = mlir::TypeAttr::get(ty);
    mlir::Value t = rewriter.create<GenTypeDescOp>(loc, tydesc, tyattr);
    mlir::Type selty = BoxType::get(rewriter.getNoneType());
    mlir::Value csel = rewriter.create<ConvertOp>(loc, selty, selector);
    mlir::Type tty = ReferenceType::get(rewriter.getNoneType());
    mlir::Value ct = rewriter.create<ConvertOp>(loc, tty, t);
    std::vector<mlir::Value> actuals = {csel, ct};
    auto fty = rewriter.getI1Type();
    std::vector<mlir::Type> argTy = {selty, tty};
    llvm::StringRef funName =
        exactTest ? "FIXME_exact_type_match" : "FIXME_isa_type_test";
    createFuncOp(rewriter.getUnknownLoc(), module, funName,
                 rewriter.getFunctionType(argTy, fty));
    // FIXME: need to call actual runtime routines for (1) testing if the
    // runtime type of the selector is an exact match to a derived type or (2)
    // testing if the runtime type of the selector is a derived type or one of
    // that derived type's subtypes.
    auto cmp = rewriter.create<mlir::CallOp>(
        loc, fty, rewriter.getSymbolRefAttr(funName), actuals);
    auto *thisBlock = rewriter.getInsertionBlock();
    auto *newBlock = createBlock(rewriter, dest);
    rewriter.setInsertionPointToEnd(thisBlock);
    if (destOps.hasValue())
      rewriter.create<mlir::CondBranchOp>(loc, cmp.getResult(0), dest,
                                          destOps.getValue(), newBlock,
                                          llvm::None);
    else
      rewriter.create<mlir::CondBranchOp>(loc, cmp.getResult(0), dest,
                                          newBlock);
    rewriter.setInsertionPointToEnd(newBlock);
  }
};

/// Convert affine dialect, fir.select_type to standard dialect
class FIRToStdLoweringPass
    : public mlir::PassWrapper<FIRToStdLoweringPass, mlir::FunctionPass> {
public:
  explicit FIRToStdLoweringPass(const KindMapping &kindMap)
      : kindMap{kindMap} {}

  void runOnFunction() override {
    if (disableFirToStd)
      return;

    // FIRToStdTypeConverter tyConv{kindMap};
    mlir::OwningRewritePatternList patterns;
    // patterns.insert<SelectTypeOpConversion>(context, tyConv);
    patterns.insert<SelectTypeOpConversion>(&getContext());
    mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
    // mlir::populateFuncOpTypeConversionPattern(patterns, context, tyConv);
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<FIROpsDialect, mlir::loop::LoopOpsDialect,
                           mlir::StandardOpsDialect>();
    // target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    //  return tyConv.isSignatureLegal(op.getType());
    //});
    target.addIllegalOp<SelectTypeOp>();

    if (mlir::failed(
            mlir::applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }

  mlir::ModuleOp getModule() {
    return getFunction().getParentOfType<mlir::ModuleOp>();
  }

private:
  const KindMapping &kindMap;
};
} // namespace

std::unique_ptr<mlir::Pass> createFIRToStdPass(const KindMapping &kindMap) {
  return std::make_unique<FIRToStdLoweringPass>(kindMap);
}
} // namespace fir
