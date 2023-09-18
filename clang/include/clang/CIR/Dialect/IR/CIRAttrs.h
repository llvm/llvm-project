//===- CIRAttrs.h - MLIR CIR Attrs ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CIR_IR_CIRATTRS_H_
#define MLIR_DIALECT_CIR_IR_CIRATTRS_H_

#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

#include "llvm/ADT/SmallVector.h"

#include "clang/CIR/Dialect/IR/CIROpsEnums.h"

#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"

//===----------------------------------------------------------------------===//
// CIR Dialect Attrs
//===----------------------------------------------------------------------===//

namespace clang {
class FunctionDecl;
class VarDecl;
class RecordDecl;
} // namespace clang

namespace mlir {
namespace cir {
class ArrayType;
class StructType;
class BoolType;
} // namespace cir
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsAttributes.h.inc"

#endif // MLIR_DIALECT_CIR_IR_CIRATTRS_H_
