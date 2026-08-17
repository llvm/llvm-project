//===- ComplexLowering.cpp - Expand complex multiply and divide -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that replaces cir.complex.mul and
// cir.complex.div with the arithmetic each one expands to, which for the full
// complex range is a call to a runtime helper such as __mulsc3 or __divsc3.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/CIRTransformUtils.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_COMPLEXLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {
struct ComplexLoweringPass
    : public impl::ComplexLoweringBase<ComplexLoweringPass> {
  ComplexLoweringPass() = default;

  void runOnOperation() override;

  void lowerComplexDivOp(cir::ComplexDivOp op);
  void lowerComplexMulOp(cir::ComplexMulOp op);

  void setASTContext(clang::ASTContext *c) { astCtx = c; }

  /// Read by the promoted-range division path, which asks the target for the
  /// semantics of a higher-precision element type.
  clang::ASTContext *astCtx = nullptr;

  mlir::ModuleOp mlirModule;
};

} // namespace

static mlir::Value buildComplexBinOpLibCall(
    mlir::ModuleOp mlirModule, CIRBaseBuilderTy &builder,
    llvm::StringRef (*libFuncNameGetter)(llvm::APFloat::Semantics),
    mlir::Location loc, cir::ComplexType ty, mlir::Value lhsReal,
    mlir::Value lhsImag, mlir::Value rhsReal, mlir::Value rhsImag) {
  cir::FPTypeInterface elementTy =
      mlir::cast<cir::FPTypeInterface>(ty.getElementType());

  llvm::StringRef libFuncName = libFuncNameGetter(
      llvm::APFloat::SemanticsToEnum(elementTy.getFloatSemantics()));
  llvm::SmallVector<mlir::Type, 4> libFuncInputTypes(4, elementTy);

  cir::FuncType libFuncTy = cir::FuncType::get(libFuncInputTypes, ty);

  // Insert a declaration for the runtime function to be used in Complex
  // multiplication and division when needed
  cir::FuncOp libFunc;
  {
    mlir::OpBuilder::InsertionGuard ipGuard{builder};
    builder.setInsertionPointToStart(mlirModule.getBody());
    libFunc = cir::buildRuntimeFunction(builder, mlirModule, libFuncName, loc,
                                        libFuncTy);
  }

  cir::CallOp call =
      builder.createCallOp(loc, libFunc, {lhsReal, lhsImag, rhsReal, rhsImag});
  return call.getResult();
}

static llvm::StringRef
getComplexDivLibCallName(llvm::APFloat::Semantics semantics) {
  switch (semantics) {
  case llvm::APFloat::S_IEEEhalf:
    return "__divhc3";
  case llvm::APFloat::S_IEEEsingle:
    return "__divsc3";
  case llvm::APFloat::S_IEEEdouble:
    return "__divdc3";
  case llvm::APFloat::S_PPCDoubleDouble:
    return "__divtc3";
  case llvm::APFloat::S_x87DoubleExtended:
    return "__divxc3";
  case llvm::APFloat::S_IEEEquad:
    return "__divtc3";
  default:
    llvm_unreachable("unsupported floating point type");
  }
}

static mlir::Value
buildAlgebraicComplexDiv(CIRBaseBuilderTy &builder, mlir::Location loc,
                         mlir::Value lhsReal, mlir::Value lhsImag,
                         mlir::Value rhsReal, mlir::Value rhsImag) {
  // (a+bi) / (c+di) = ((ac+bd)/(cc+dd)) + ((bc-ad)/(cc+dd))i
  mlir::Value &a = lhsReal;
  mlir::Value &b = lhsImag;
  mlir::Value &c = rhsReal;
  mlir::Value &d = rhsImag;

  // The element type of the complex (lhs/rhs) determines whether floating
  // point or integer ops are needed.
  bool isFP = cir::isFPOrVectorOfFPType(a.getType());
  auto mul = [&](mlir::Location l, mlir::Value x, mlir::Value y) {
    return isFP ? builder.createFMul(l, x, y) : builder.createMul(l, x, y);
  };
  auto add = [&](mlir::Location l, mlir::Value x, mlir::Value y) {
    return isFP ? builder.createFAdd(l, x, y) : builder.createAdd(l, x, y);
  };
  auto sub = [&](mlir::Location l, mlir::Value x, mlir::Value y) {
    return isFP ? builder.createFSub(l, x, y) : builder.createSub(l, x, y);
  };
  auto div = [&](mlir::Location l, mlir::Value x, mlir::Value y) {
    return isFP ? builder.createFDiv(l, x, y) : builder.createDiv(l, x, y);
  };

  mlir::Value ac = mul(loc, a, c);     // a*c
  mlir::Value bd = mul(loc, b, d);     // b*d
  mlir::Value cc = mul(loc, c, c);     // c*c
  mlir::Value dd = mul(loc, d, d);     // d*d
  mlir::Value acbd = add(loc, ac, bd); // ac+bd
  mlir::Value ccdd = add(loc, cc, dd); // cc+dd
  mlir::Value resultReal = div(loc, acbd, ccdd);

  mlir::Value bc = mul(loc, b, c);     // b*c
  mlir::Value ad = mul(loc, a, d);     // a*d
  mlir::Value bcad = sub(loc, bc, ad); // bc-ad
  mlir::Value resultImag = div(loc, bcad, ccdd);
  return builder.createComplexCreate(loc, resultReal, resultImag);
}

static mlir::Value
buildRangeReductionComplexDiv(CIRBaseBuilderTy &builder, mlir::Location loc,
                              mlir::Value lhsReal, mlir::Value lhsImag,
                              mlir::Value rhsReal, mlir::Value rhsImag) {
  // Implements Smith's algorithm for complex division.
  // SMITH, R. L. Algorithm 116: Complex division. Commun. ACM 5, 8 (1962).

  // Let:
  //   - lhs := a+bi
  //   - rhs := c+di
  //   - result := lhs / rhs = e+fi
  //
  // The algorithm pseudocode looks like follows:
  //   if fabs(c) >= fabs(d):
  //     r := d / c
  //     tmp := c + r*d
  //     e = (a + b*r) / tmp
  //     f = (b - a*r) / tmp
  //   else:
  //     r := c / d
  //     tmp := d + r*c
  //     e = (a*r + b) / tmp
  //     f = (b*r - a) / tmp

  mlir::Value &a = lhsReal;
  mlir::Value &b = lhsImag;
  mlir::Value &c = rhsReal;
  mlir::Value &d = rhsImag;

  // Smith's algorithm is only used for floating-point complex division.
  assert(cir::isFPOrVectorOfFPType(a.getType()) &&
         "range-reduction complex divide expects floating-point operands");

  auto trueBranchBuilder = [&](mlir::OpBuilder &, mlir::Location) {
    mlir::Value r = builder.createFDiv(loc, d, c);    // r := d / c
    mlir::Value rd = builder.createFMul(loc, r, d);   // r*d
    mlir::Value tmp = builder.createFAdd(loc, c, rd); // tmp := c + r*d

    mlir::Value br = builder.createFMul(loc, b, r);   // b*r
    mlir::Value abr = builder.createFAdd(loc, a, br); // a + b*r
    mlir::Value e = builder.createFDiv(loc, abr, tmp);

    mlir::Value ar = builder.createFMul(loc, a, r);   // a*r
    mlir::Value bar = builder.createFSub(loc, b, ar); // b - a*r
    mlir::Value f = builder.createFDiv(loc, bar, tmp);

    mlir::Value result = builder.createComplexCreate(loc, e, f);
    builder.createYield(loc, result);
  };

  auto falseBranchBuilder = [&](mlir::OpBuilder &, mlir::Location) {
    mlir::Value r = builder.createFDiv(loc, c, d);    // r := c / d
    mlir::Value rc = builder.createFMul(loc, r, c);   // r*c
    mlir::Value tmp = builder.createFAdd(loc, d, rc); // tmp := d + r*c

    mlir::Value ar = builder.createFMul(loc, a, r);   // a*r
    mlir::Value arb = builder.createFAdd(loc, ar, b); // a*r + b
    mlir::Value e = builder.createFDiv(loc, arb, tmp);

    mlir::Value br = builder.createFMul(loc, b, r);   // b*r
    mlir::Value bra = builder.createFSub(loc, br, a); // b*r - a
    mlir::Value f = builder.createFDiv(loc, bra, tmp);

    mlir::Value result = builder.createComplexCreate(loc, e, f);
    builder.createYield(loc, result);
  };

  auto cFabs = cir::FAbsOp::create(builder, loc, c);
  auto dFabs = cir::FAbsOp::create(builder, loc, d);
  cir::CmpOp cmpResult =
      builder.createCompare(loc, cir::CmpOpKind::ge, cFabs, dFabs);
  auto ternary = cir::TernaryOp::create(builder, loc, cmpResult,
                                        trueBranchBuilder, falseBranchBuilder);

  return ternary.getResult();
}

static mlir::Type higherPrecisionElementTypeForComplexArithmetic(
    mlir::MLIRContext &context, clang::ASTContext &cc,
    CIRBaseBuilderTy &builder, mlir::Type elementType) {

  auto getHigherPrecisionFPType = [&context](mlir::Type type) -> mlir::Type {
    if (mlir::isa<cir::FP16Type>(type))
      return cir::SingleType::get(&context);

    if (mlir::isa<cir::SingleType>(type) || mlir::isa<cir::BF16Type>(type))
      return cir::DoubleType::get(&context);

    if (mlir::isa<cir::DoubleType>(type))
      return cir::LongDoubleType::get(&context, type);

    return type;
  };

  auto getFloatTypeSemantics =
      [&cc](mlir::Type type) -> const llvm::fltSemantics & {
    const clang::TargetInfo &info = cc.getTargetInfo();
    if (mlir::isa<cir::FP16Type>(type))
      return info.getHalfFormat();

    if (mlir::isa<cir::BF16Type>(type))
      return info.getBFloat16Format();

    if (mlir::isa<cir::SingleType>(type))
      return info.getFloatFormat();

    if (mlir::isa<cir::DoubleType>(type))
      return info.getDoubleFormat();

    if (mlir::isa<cir::LongDoubleType>(type)) {
      if (cc.getLangOpts().OpenMP && cc.getLangOpts().OpenMPIsTargetDevice)
        llvm_unreachable("NYI Float type semantics with OpenMP");
      return info.getLongDoubleFormat();
    }

    if (mlir::isa<cir::FP128Type>(type)) {
      if (cc.getLangOpts().OpenMP && cc.getLangOpts().OpenMPIsTargetDevice)
        llvm_unreachable("NYI Float type semantics with OpenMP");
      return info.getFloat128Format();
    }

    llvm_unreachable("Unsupported float type semantics");
  };

  const mlir::Type higherElementType = getHigherPrecisionFPType(elementType);
  const llvm::fltSemantics &elementTypeSemantics =
      getFloatTypeSemantics(elementType);
  const llvm::fltSemantics &higherElementTypeSemantics =
      getFloatTypeSemantics(higherElementType);

  // Check that the promoted type can handle the intermediate values without
  // overflowing. This can be interpreted as:
  // (SmallerType.LargestFiniteVal * SmallerType.LargestFiniteVal) * 2 <=
  //      LargerType.LargestFiniteVal.
  // In terms of exponent it gives this formula:
  // (SmallerType.LargestFiniteVal * SmallerType.LargestFiniteVal
  // doubles the exponent of SmallerType.LargestFiniteVal)
  if (llvm::APFloat::semanticsMaxExponent(elementTypeSemantics) * 2 + 1 <=
      llvm::APFloat::semanticsMaxExponent(higherElementTypeSemantics)) {
    return higherElementType;
  }

  // The intermediate values can't be represented in the promoted type
  // without overflowing.
  return {};
}

static mlir::Value
lowerComplexDiv(mlir::ModuleOp mlirModule, CIRBaseBuilderTy &builder,
                mlir::Location loc, cir::ComplexDivOp op, mlir::Value lhsReal,
                mlir::Value lhsImag, mlir::Value rhsReal, mlir::Value rhsImag,
                mlir::MLIRContext &mlirCx, clang::ASTContext &cc) {
  cir::ComplexType complexTy = op.getType();
  if (mlir::isa<cir::FPTypeInterface>(complexTy.getElementType())) {
    cir::ComplexRangeKind range = op.getRange();
    if (range == cir::ComplexRangeKind::Improved)
      return buildRangeReductionComplexDiv(builder, loc, lhsReal, lhsImag,
                                           rhsReal, rhsImag);

    if (range == cir::ComplexRangeKind::Full)
      return buildComplexBinOpLibCall(mlirModule, builder,
                                      &getComplexDivLibCallName, loc, complexTy,
                                      lhsReal, lhsImag, rhsReal, rhsImag);

    if (range == cir::ComplexRangeKind::Promoted) {
      mlir::Type originalElementType = complexTy.getElementType();
      mlir::Type higherPrecisionElementType =
          higherPrecisionElementTypeForComplexArithmetic(mlirCx, cc, builder,
                                                         originalElementType);

      if (!higherPrecisionElementType)
        return buildRangeReductionComplexDiv(builder, loc, lhsReal, lhsImag,
                                             rhsReal, rhsImag);

      cir::CastKind floatingCastKind = cir::CastKind::floating;
      lhsReal = builder.createCast(floatingCastKind, lhsReal,
                                   higherPrecisionElementType);
      lhsImag = builder.createCast(floatingCastKind, lhsImag,
                                   higherPrecisionElementType);
      rhsReal = builder.createCast(floatingCastKind, rhsReal,
                                   higherPrecisionElementType);
      rhsImag = builder.createCast(floatingCastKind, rhsImag,
                                   higherPrecisionElementType);

      mlir::Value algebraicResult = buildAlgebraicComplexDiv(
          builder, loc, lhsReal, lhsImag, rhsReal, rhsImag);

      mlir::Value resultReal = builder.createComplexReal(loc, algebraicResult);
      mlir::Value resultImag = builder.createComplexImag(loc, algebraicResult);

      mlir::Value finalReal =
          builder.createCast(floatingCastKind, resultReal, originalElementType);
      mlir::Value finalImag =
          builder.createCast(floatingCastKind, resultImag, originalElementType);
      return builder.createComplexCreate(loc, finalReal, finalImag);
    }
  }

  return buildAlgebraicComplexDiv(builder, loc, lhsReal, lhsImag, rhsReal,
                                  rhsImag);
}

void ComplexLoweringPass::lowerComplexDivOp(cir::ComplexDivOp op) {
  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);
  mlir::Location loc = op.getLoc();
  mlir::TypedValue<cir::ComplexType> lhs = op.getLhs();
  mlir::TypedValue<cir::ComplexType> rhs = op.getRhs();
  mlir::Value lhsReal = builder.createComplexReal(loc, lhs);
  mlir::Value lhsImag = builder.createComplexImag(loc, lhs);
  mlir::Value rhsReal = builder.createComplexReal(loc, rhs);
  mlir::Value rhsImag = builder.createComplexImag(loc, rhs);

  mlir::Value loweredResult =
      lowerComplexDiv(mlirModule, builder, loc, op, lhsReal, lhsImag, rhsReal,
                      rhsImag, getContext(), *astCtx);
  op.replaceAllUsesWith(loweredResult);
  op.erase();
}

static llvm::StringRef
getComplexMulLibCallName(llvm::APFloat::Semantics semantics) {
  switch (semantics) {
  case llvm::APFloat::S_IEEEhalf:
    return "__mulhc3";
  case llvm::APFloat::S_IEEEsingle:
    return "__mulsc3";
  case llvm::APFloat::S_IEEEdouble:
    return "__muldc3";
  case llvm::APFloat::S_PPCDoubleDouble:
    return "__multc3";
  case llvm::APFloat::S_x87DoubleExtended:
    return "__mulxc3";
  case llvm::APFloat::S_IEEEquad:
    return "__multc3";
  default:
    llvm_unreachable("unsupported floating point type");
  }
}

static mlir::Value lowerComplexMul(mlir::ModuleOp mlirModule,
                                   CIRBaseBuilderTy &builder,
                                   mlir::Location loc, cir::ComplexMulOp op,
                                   mlir::Value lhsReal, mlir::Value lhsImag,
                                   mlir::Value rhsReal, mlir::Value rhsImag) {
  // (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
  bool isFP = cir::isFPOrVectorOfFPType(lhsReal.getType());
  auto mul = [&](mlir::Location l, mlir::Value x, mlir::Value y) {
    return isFP ? builder.createFMul(l, x, y) : builder.createMul(l, x, y);
  };
  auto add = [&](mlir::Location l, mlir::Value x, mlir::Value y) {
    return isFP ? builder.createFAdd(l, x, y) : builder.createAdd(l, x, y);
  };
  auto sub = [&](mlir::Location l, mlir::Value x, mlir::Value y) {
    return isFP ? builder.createFSub(l, x, y) : builder.createSub(l, x, y);
  };

  mlir::Value resultRealLhs = mul(loc, lhsReal, rhsReal); // ac
  mlir::Value resultRealRhs = mul(loc, lhsImag, rhsImag); // bd
  mlir::Value resultImagLhs = mul(loc, lhsReal, rhsImag); // ad
  mlir::Value resultImagRhs = mul(loc, lhsImag, rhsReal); // bc
  mlir::Value resultReal = sub(loc, resultRealLhs, resultRealRhs);
  mlir::Value resultImag = add(loc, resultImagLhs, resultImagRhs);
  mlir::Value algebraicResult =
      builder.createComplexCreate(loc, resultReal, resultImag);

  cir::ComplexType complexTy = op.getType();
  cir::ComplexRangeKind rangeKind = op.getRange();
  if (mlir::isa<cir::IntType>(complexTy.getElementType()) ||
      rangeKind == cir::ComplexRangeKind::Basic ||
      rangeKind == cir::ComplexRangeKind::Improved ||
      rangeKind == cir::ComplexRangeKind::Promoted)
    return algebraicResult;

  assert(!cir::MissingFeatures::fastMathFlags());

  // Check whether the real part and the imaginary part of the result are both
  // NaN. If so, emit a library call to compute the multiplication instead.
  // We check a value against NaN by comparing the value against itself.
  mlir::Value resultRealIsNaN = builder.createIsNaN(loc, resultReal);
  mlir::Value resultImagIsNaN = builder.createIsNaN(loc, resultImag);
  mlir::Value resultRealAndImagAreNaN =
      builder.createLogicalAnd(loc, resultRealIsNaN, resultImagIsNaN);

  return cir::TernaryOp::create(
             builder, loc, resultRealAndImagAreNaN,
             [&](mlir::OpBuilder &, mlir::Location) {
               mlir::Value libCallResult = buildComplexBinOpLibCall(
                   mlirModule, builder, &getComplexMulLibCallName, loc,
                   complexTy, lhsReal, lhsImag, rhsReal, rhsImag);
               builder.createYield(loc, libCallResult);
             },
             [&](mlir::OpBuilder &, mlir::Location) {
               builder.createYield(loc, algebraicResult);
             })
      .getResult();
}

void ComplexLoweringPass::lowerComplexMulOp(cir::ComplexMulOp op) {
  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);
  mlir::Location loc = op.getLoc();
  mlir::TypedValue<cir::ComplexType> lhs = op.getLhs();
  mlir::TypedValue<cir::ComplexType> rhs = op.getRhs();
  mlir::Value lhsReal = builder.createComplexReal(loc, lhs);
  mlir::Value lhsImag = builder.createComplexImag(loc, lhs);
  mlir::Value rhsReal = builder.createComplexReal(loc, rhs);
  mlir::Value rhsImag = builder.createComplexImag(loc, rhs);
  mlir::Value loweredResult = lowerComplexMul(
      mlirModule, builder, loc, op, lhsReal, lhsImag, rhsReal, rhsImag);
  op.replaceAllUsesWith(loweredResult);
  op.erase();
}

void ComplexLoweringPass::runOnOperation() {
  assert(astCtx && "complex lowering requires an ASTContext");
  mlirModule = mlir::cast<mlir::ModuleOp>(getOperation());

  llvm::SmallVector<mlir::Operation *> opsToTransform;
  mlirModule->walk([&](mlir::Operation *op) {
    if (mlir::isa<cir::ComplexMulOp, cir::ComplexDivOp>(op))
      opsToTransform.push_back(op);
  });

  for (mlir::Operation *o : opsToTransform) {
    if (auto complexDiv = mlir::dyn_cast<cir::ComplexDivOp>(o))
      lowerComplexDivOp(complexDiv);
    else
      lowerComplexMulOp(mlir::cast<cir::ComplexMulOp>(o));
  }
}

std::unique_ptr<Pass> mlir::createComplexLoweringPass() {
  return std::make_unique<ComplexLoweringPass>();
}

std::unique_ptr<Pass>
mlir::createComplexLoweringPass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<ComplexLoweringPass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
