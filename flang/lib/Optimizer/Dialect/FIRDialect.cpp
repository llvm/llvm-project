//===-- FIRDialect.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace fir;

namespace {
/// This class defines the interface for handling inlining of FIR calls.
struct FIRInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    return fir::canLegallyInline(call, callable, wouldBeCloned);
  }

  /// This hook checks to see if the operation `op` is legal to inline into the
  /// given region `reg`.
  bool isLegalToInline(mlir::Operation *op, mlir::Region *reg,
                       bool wouldBeCloned,
                       mlir::BlockAndValueMapping &map) const final {
    return fir::canLegallyInline(op, reg, wouldBeCloned, map);
  }

  /// This hook is called when a terminator operation has been inlined.
  /// We handle the return (a Fortran FUNCTION) by replacing the values
  /// previously returned by the call operation with the operands of the
  /// return.
  void handleTerminator(mlir::Operation *op,
                        llvm::ArrayRef<mlir::Value> valuesToRepl) const final {
    auto returnOp = cast<mlir::ReturnOp>(op);
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  mlir::Operation *materializeCallConversion(mlir::OpBuilder &builder,
                                             mlir::Value input,
                                             mlir::Type resultType,
                                             mlir::Location loc) const final {
    return builder.create<fir::ConvertOp>(loc, resultType, input);
  }
};
} // namespace

fir::FIROpsDialect::FIROpsDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect("fir", ctx, mlir::TypeID::get<FIROpsDialect>()) {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "flang/Optimizer/Dialect/FIROps.cpp.inc"
      >();
  addInterfaces<FIRInlinerInterface>();
}

// anchor the class vtable to this compilation unit
fir::FIROpsDialect::~FIROpsDialect() {
  // do nothing
}

mlir::Type fir::FIROpsDialect::parseType(mlir::DialectAsmParser &parser) const {
  return parseFirType(const_cast<FIROpsDialect *>(this), parser);
}

void fir::FIROpsDialect::printType(mlir::Type ty,
                                   mlir::DialectAsmPrinter &p) const {
  return printFirType(const_cast<FIROpsDialect *>(this), ty, p);
}

mlir::Attribute
fir::FIROpsDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                   mlir::Type type) const {
  return parseFirAttribute(const_cast<FIROpsDialect *>(this), parser, type);
}

void fir::FIROpsDialect::printAttribute(mlir::Attribute attr,
                                        mlir::DialectAsmPrinter &p) const {
  printFirAttribute(const_cast<FIROpsDialect *>(this), attr, p);
}
