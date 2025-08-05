//====- LoweringHelpers.h - Lowering helper functions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helper functions for lowering from CIR to LLVM or MLIR.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_CIR_LOWERINGHELPERS_H
#define LLVM_CLANG_CIR_LOWERINGHELPERS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

mlir::DenseElementsAttr
convertStringAttrToDenseElementsAttr(cir::ConstArrayAttr attr, mlir::Type type);

template <typename StorageTy> StorageTy getZeroInitFromType(mlir::Type ty);
template <> mlir::APInt getZeroInitFromType(mlir::Type ty);
template <> mlir::APFloat getZeroInitFromType(mlir::Type ty);

template <typename AttrTy, typename StorageTy>
void convertToDenseElementsAttrImpl(cir::ConstArrayAttr attr,
                                    llvm::SmallVectorImpl<StorageTy> &values);

template <typename AttrTy, typename StorageTy>
mlir::DenseElementsAttr
convertToDenseElementsAttr(cir::ConstArrayAttr attr,
                           const llvm::SmallVectorImpl<int64_t> &dims,
                           mlir::Type type);

std::optional<mlir::Attribute>
lowerConstArrayAttr(cir::ConstArrayAttr constArr,
                    const mlir::TypeConverter *converter);

mlir::Value getConstAPInt(mlir::OpBuilder &bld, mlir::Location loc,
                          mlir::Type typ, const llvm::APInt &val);

mlir::Value getConst(mlir::OpBuilder &bld, mlir::Location loc, mlir::Type typ,
                     unsigned val);

mlir::Value createShL(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs);

mlir::Value createAShR(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs);

mlir::Value createAnd(mlir::OpBuilder &bld, mlir::Value lhs,
                      const llvm::APInt &rhs);

mlir::Value createLShR(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs);
#endif
