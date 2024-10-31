//===- CIRDialect.cpp - MLIR CIR ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "mlir/Support/LogicalResult.h"

#include "clang/CIR/Dialect/IR/CIROpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//

void mlir::cir::CIRDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "clang/CIR/Dialect/IR/CIROps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void mlir::cir::FuncOp::build(OpBuilder &builder, OperationState &result,
                              StringRef name) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
}

ParseResult cir::FuncOp::parse(OpAsmParser &parser, OperationState &state) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return failure();
  return success();
}

void cir::FuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  // For now the only property a function has is its name
  p.printSymbolName(getSymName());
}

mlir::LogicalResult mlir::cir::FuncOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROps.cpp.inc"
