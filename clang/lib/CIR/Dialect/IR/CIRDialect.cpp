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

#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "mlir/Support/LogicalResult.h"

#include "clang/CIR/Dialect/IR/CIROpsDialect.cpp.inc"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//

void cir::CIRDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "clang/CIR/Dialect/IR/CIROps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult checkConstantTypes(mlir::Operation *op, mlir::Type opType,
                                        mlir::Attribute attrType) {
  if (isa<cir::ConstPtrAttr>(attrType)) {
    if (!mlir::isa<cir::PointerType>(opType))
      return op->emitOpError(
          "pointer constant initializing a non-pointer type");
    return success();
  }

  if (mlir::isa<cir::IntAttr, cir::FPAttr>(attrType)) {
    auto at = cast<TypedAttr>(attrType);
    if (at.getType() != opType) {
      return op->emitOpError("result type (")
             << opType << ") does not match value type (" << at.getType()
             << ")";
    }
    return success();
  }

  assert(isa<TypedAttr>(attrType) && "What else could we be looking at here?");
  return op->emitOpError("global with type ")
         << cast<TypedAttr>(attrType).getType() << " not yet supported";
}

LogicalResult cir::ConstantOp::verify() {
  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  return checkConstantTypes(getOperation(), getType(), getValue());
}

OpFoldResult cir::ConstantOp::fold(FoldAdaptor /*adaptor*/) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static ParseResult parseConstantValue(OpAsmParser &parser,
                                      mlir::Attribute &valueAttr) {
  NamedAttrList attr;
  return parser.parseAttribute(valueAttr, "value", attr);
}

static void printConstant(OpAsmPrinter &p, Attribute value) {
  p.printAttribute(value);
}

mlir::LogicalResult cir::GlobalOp::verify() {
  // Verify that the initial value, if present, is either a unit attribute or
  // an attribute CIR supports.
  if (getInitialValue().has_value()) {
    if (checkConstantTypes(getOperation(), getSymType(), *getInitialValue())
            .failed())
      return failure();
  }

  // TODO(CIR): Many other checks for properties that haven't been upstreamed
  // yet.

  return success();
}

void cir::GlobalOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          llvm::StringRef sym_name, mlir::Type sym_type) {
  odsState.addAttribute(getSymNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getSymTypeAttrName(odsState.name),
                        mlir::TypeAttr::get(sym_type));
}

static void printGlobalOpTypeAndInitialValue(OpAsmPrinter &p, cir::GlobalOp op,
                                             TypeAttr type,
                                             Attribute initAttr) {
  if (!op.isDeclaration()) {
    p << "= ";
    // This also prints the type...
    if (initAttr)
      printConstant(p, initAttr);
  } else {
    p << ": " << type;
  }
}

static ParseResult
parseGlobalOpTypeAndInitialValue(OpAsmParser &parser, TypeAttr &typeAttr,
                                 Attribute &initialValueAttr) {
  mlir::Type opTy;
  if (parser.parseOptionalEqual().failed()) {
    // Absence of equal means a declaration, so we need to parse the type.
    //  cir.global @a : !cir.int<s, 32>
    if (parser.parseColonType(opTy))
      return failure();
  } else {
    // Parse constant with initializer, examples:
    //  cir.global @y = #cir.fp<1.250000e+00> : !cir.double
    //  cir.global @rgb = #cir.const_array<[...] : !cir.array<i8 x 3>>
    if (parseConstantValue(parser, initialValueAttr).failed())
      return failure();

    assert(mlir::isa<mlir::TypedAttr>(initialValueAttr) &&
           "Non-typed attrs shouldn't appear here.");
    auto typedAttr = mlir::cast<mlir::TypedAttr>(initialValueAttr);
    opTy = typedAttr.getType();
  }

  typeAttr = TypeAttr::get(opTy);
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void cir::FuncOp::build(OpBuilder &builder, OperationState &result,
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

// TODO(CIR): The properties of functions that require verification haven't
// been implemented yet.
mlir::LogicalResult cir::FuncOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROps.cpp.inc"
