//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CIR_IR_CIRTYPES_H_
#define MLIR_DIALECT_CIR_IR_CIRTYPES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Interfaces/CIRFPTypeInterface.h"

namespace cir {

bool isAnyFloatingPointType(mlir::Type t);

} // namespace cir

//===----------------------------------------------------------------------===//
// CIR Dialect Tablegen'd Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsTypes.h.inc"

#endif // MLIR_DIALECT_CIR_IR_CIRTYPES_H_
