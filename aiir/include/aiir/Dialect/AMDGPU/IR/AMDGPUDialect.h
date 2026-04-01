//===- AMDGPUDialect.h - AIIR Dialect for AMDGPU ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a dialect for AIIR wrappers around AMDGPU-specific
// intrinsics and for other AMD GPU-specific functionality.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_AMDGPU_IR_AMDGPUDIALECT_H_
#define AIIR_DIALECT_AMDGPU_IR_AMDGPUDIALECT_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/ViewLikeInterface.h"

#include "aiir/Dialect/AMDGPU/IR/AMDGPUDialect.h.inc"

#include "aiir/Dialect/AMDGPU/IR/AMDGPUEnums.h.inc"

#include "aiir/Dialect/AMDGPU/IR/AMDGPUAttrs.h.inc"
#include "aiir/Dialect/AMDGPU/IR/AMDGPUTypes.h.inc"

namespace aiir::amdgpu {
/// Parser for the `custom<MNKDimensionList>` custom assembly format used by
/// WMMAOp.
ParseResult parseMNKDimensionList(OpAsmParser &parser, IntegerAttr &m,
                                  IntegerAttr &n, IntegerAttr &k);
inline ParseResult parseMNKDimensionList(OpAsmParser &parser, Operation *,
                                         IntegerAttr &m, IntegerAttr &n,
                                         IntegerAttr &k) {
  return parseMNKDimensionList(parser, m, n, k);
}

/// Printer for the `custom<MNKDimensionList>` custom assembly format used by
/// WMMAOp.
inline void printMNKDimensionList(OpAsmPrinter &printer, IntegerAttr m,
                                  IntegerAttr n, IntegerAttr k) {
  printer.printDimensionList(ArrayRef{m.getInt(), n.getInt(), k.getInt()});
}
inline void printMNKDimensionList(OpAsmPrinter &printer, Operation *,
                                  IntegerAttr m, IntegerAttr n, IntegerAttr k) {
  printMNKDimensionList(printer, m, n, k);
}
} // namespace aiir::amdgpu

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/AMDGPU/IR/AMDGPUAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/AMDGPU/IR/AMDGPUTypes.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/AMDGPU/IR/AMDGPU.h.inc"

#endif // AIIR_DIALECT_AMDGPU_IR_AMDGPUDIALECT_H_
