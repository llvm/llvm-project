//====- LoweringHelpers.h - Lowering helper functions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helper functions for lowering from CIR to LLVM or AIIR.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_CIR_LOWERINGHELPERS_H
#define LLVM_CLANG_CIR_LOWERINGHELPERS_H

#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

aiir::DenseElementsAttr
convertStringAttrToDenseElementsAttr(cir::ConstArrayAttr attr, aiir::Type type);

template <typename StorageTy> StorageTy getZeroInitFromType(aiir::Type ty);
template <> aiir::APInt getZeroInitFromType(aiir::Type ty);
template <> aiir::APFloat getZeroInitFromType(aiir::Type ty);

template <typename AttrTy, typename StorageTy>
void convertToDenseElementsAttrImpl(cir::ConstArrayAttr attr,
                                    llvm::SmallVectorImpl<StorageTy> &values);

template <typename AttrTy, typename StorageTy>
aiir::DenseElementsAttr
convertToDenseElementsAttr(cir::ConstArrayAttr attr,
                           const llvm::SmallVectorImpl<int64_t> &dims,
                           aiir::Type type);

std::optional<aiir::Attribute>
lowerConstArrayAttr(cir::ConstArrayAttr constArr,
                    const aiir::TypeConverter *converter);

aiir::Value getConstAPInt(aiir::OpBuilder &bld, aiir::Location loc,
                          aiir::Type typ, const llvm::APInt &val);

aiir::Value getConst(aiir::OpBuilder &bld, aiir::Location loc, aiir::Type typ,
                     unsigned val);

aiir::Value createShL(aiir::OpBuilder &bld, aiir::Value lhs, unsigned rhs);

aiir::Value createAShR(aiir::OpBuilder &bld, aiir::Value lhs, unsigned rhs);

aiir::Value createAnd(aiir::OpBuilder &bld, aiir::Value lhs,
                      const llvm::APInt &rhs);

aiir::Value createLShR(aiir::OpBuilder &bld, aiir::Value lhs, unsigned rhs);
#endif
