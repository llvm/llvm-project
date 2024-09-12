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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

mlir::DenseElementsAttr
convertStringAttrToDenseElementsAttr(mlir::cir::ConstArrayAttr attr,
                                     mlir::Type type);

template <typename StorageTy> StorageTy getZeroInitFromType(mlir::Type Ty);
template <> mlir::APInt getZeroInitFromType(mlir::Type Ty);
template <> mlir::APFloat getZeroInitFromType(mlir::Type Ty);

mlir::Type getNestedTypeAndElemQuantity(mlir::Type Ty, unsigned &elemQuantity);

template <typename AttrTy, typename StorageTy>
void convertToDenseElementsAttrImpl(mlir::cir::ConstArrayAttr attr,
                                    llvm::SmallVectorImpl<StorageTy> &values);

template <typename AttrTy, typename StorageTy>
mlir::DenseElementsAttr
convertToDenseElementsAttr(mlir::cir::ConstArrayAttr attr,
                           const llvm::SmallVectorImpl<int64_t> &dims,
                           mlir::Type type);

std::optional<mlir::Attribute>
lowerConstArrayAttr(mlir::cir::ConstArrayAttr constArr,
                    const mlir::TypeConverter *converter);
#endif
