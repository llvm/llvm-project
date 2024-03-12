#ifndef LLVM_CLANG_LIB_LOWERINGHELPERS_H
#define LLVM_CLANG_LIB_LOWERINGHELPERS_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

using namespace llvm;

mlir::Value createIntCast(mlir::OpBuilder &bld, mlir::Value src,
                          mlir::IntegerType dstTy, bool isSigned = false) {
  auto srcTy = src.getType();
  assert(isa<mlir::IntegerType>(srcTy));

  auto srcWidth = srcTy.cast<mlir::IntegerType>().getWidth();
  auto dstWidth = dstTy.cast<mlir::IntegerType>().getWidth();
  auto loc = src.getLoc();

  if (dstWidth > srcWidth && isSigned)
    return bld.create<mlir::LLVM::SExtOp>(loc, dstTy, src);
  else if (dstWidth > srcWidth)
    return bld.create<mlir::LLVM::ZExtOp>(loc, dstTy, src);
  else if (dstWidth < srcWidth)
    return bld.create<mlir::LLVM::TruncOp>(loc, dstTy, src);
  else
    return bld.create<mlir::LLVM::BitcastOp>(loc, dstTy, src);
}

mlir::Value getConstAPInt(mlir::OpBuilder &bld, mlir::Location loc,
                          mlir::Type typ, const llvm::APInt &val) {
  return bld.create<mlir::LLVM::ConstantOp>(loc, typ, val);
}

mlir::Value getConst(mlir::OpBuilder &bld, mlir::Location loc, mlir::Type typ,
                     unsigned val) {
  return bld.create<mlir::LLVM::ConstantOp>(loc, typ, val);
}

mlir::Value createShL(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  auto rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::ShlOp>(lhs.getLoc(), lhs, rhsVal);
}

mlir::Value createLShR(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  auto rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::LShrOp>(lhs.getLoc(), lhs, rhsVal);
}

mlir::Value createAShR(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  auto rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::AShrOp>(lhs.getLoc(), lhs, rhsVal);
}

mlir::Value createAnd(mlir::OpBuilder &bld, mlir::Value lhs,
                      const llvm::APInt &rhs) {
  auto rhsVal = getConstAPInt(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::AndOp>(lhs.getLoc(), lhs, rhsVal);
}

#endif // LLVM_CLANG_LIB_LOWERINGHELPERS_H