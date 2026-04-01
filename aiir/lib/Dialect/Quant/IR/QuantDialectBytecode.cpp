//===- QuantDialectBytecode.cpp - Quant Bytecode Implementation
//------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QuantDialectBytecode.h"
#include "aiir/Bytecode/BytecodeImplementation.h"
#include "aiir/Dialect/Quant/IR/Quant.h"
#include "aiir/Dialect/Quant/IR/QuantTypes.h"
#include "aiir/IR/Diagnostics.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace aiir;
using namespace aiir::quant;

namespace {

static LogicalResult readDoubleAPFloat(DialectBytecodeReader &reader,
                                       double &val) {
  auto valOr =
      reader.readAPFloatWithKnownSemantics(llvm::APFloat::IEEEdouble());
  if (failed(valOr))
    return failure();
  val = valOr->convertToDouble();
  return success();
}

#include "aiir/Dialect/Quant/IR/QuantDialectBytecode.cpp.inc"

/// This class implements the bytecode interface for the Quant dialect.
struct QuantDialectBytecodeInterface : public BytecodeDialectInterface {
  QuantDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    return ::writeAttribute(attr, writer);
  }

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return ::writeType(type, writer);
  }
};
} // namespace

void quant::detail::addBytecodeInterface(QuantDialect *dialect) {
  dialect->addInterfaces<QuantDialectBytecodeInterface>();
}
