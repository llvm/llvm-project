#ifndef LLVM_CLANG_LIB_LOWERTOMLIRHELPERS_H
#define LLVM_CLANG_LIB_LOWERTOMLIRHELPERS_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

template <typename T>
mlir::Value getConst(mlir::ConversionPatternRewriter &rewriter,
                     mlir::Location loc, mlir::Type ty, T value) {
  assert(mlir::isa<mlir::IntegerType>(ty) || mlir::isa<mlir::FloatType>(ty));
  if (mlir::isa<mlir::IntegerType>(ty))
    return rewriter.create<mlir::arith::ConstantOp>(
        loc, ty, mlir::IntegerAttr::get(ty, value));
  return rewriter.create<mlir::arith::ConstantOp>(
      loc, ty, mlir::FloatAttr::get(ty, value));
}

mlir::Value createIntCast(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Value src, mlir::Type dstTy,
                          bool isSigned = false) {
  auto srcTy = src.getType();
  assert(mlir::isa<mlir::IntegerType>(srcTy));
  assert(mlir::isa<mlir::IntegerType>(dstTy));

  auto srcWidth = srcTy.cast<mlir::IntegerType>().getWidth();
  auto dstWidth = dstTy.cast<mlir::IntegerType>().getWidth();
  auto loc = src.getLoc();

  if (dstWidth > srcWidth && isSigned)
    return rewriter.create<mlir::arith::ExtSIOp>(loc, dstTy, src);
  else if (dstWidth > srcWidth)
    return rewriter.create<mlir::arith::ExtUIOp>(loc, dstTy, src);
  else if (dstWidth < srcWidth)
    return rewriter.create<mlir::arith::TruncIOp>(loc, dstTy, src);
  else
    return rewriter.create<mlir::arith::BitcastOp>(loc, dstTy, src);
}

mlir::arith::CmpIPredicate
convertCmpKindToCmpIPredicate(mlir::cir::CmpOpKind kind, bool isSigned) {
  using CIR = mlir::cir::CmpOpKind;
  using arithCmpI = mlir::arith::CmpIPredicate;
  switch (kind) {
  case CIR::eq:
    return arithCmpI::eq;
  case CIR::ne:
    return arithCmpI::ne;
  case CIR::lt:
    return (isSigned ? arithCmpI::slt : arithCmpI::ult);
  case CIR::le:
    return (isSigned ? arithCmpI::sle : arithCmpI::ule);
  case CIR::gt:
    return (isSigned ? arithCmpI::sgt : arithCmpI::ugt);
  case CIR::ge:
    return (isSigned ? arithCmpI::sge : arithCmpI::uge);
  }
  llvm_unreachable("Unknown CmpOpKind");
}

mlir::arith::CmpFPredicate
convertCmpKindToCmpFPredicate(mlir::cir::CmpOpKind kind) {
  using CIR = mlir::cir::CmpOpKind;
  using arithCmpF = mlir::arith::CmpFPredicate;
  switch (kind) {
  case CIR::eq:
    return arithCmpF::OEQ;
  case CIR::ne:
    return arithCmpF::UNE;
  case CIR::lt:
    return arithCmpF::OLT;
  case CIR::le:
    return arithCmpF::OLE;
  case CIR::gt:
    return arithCmpF::OGT;
  case CIR::ge:
    return arithCmpF::OGE;
  }
  llvm_unreachable("Unknown CmpOpKind");
}

#endif