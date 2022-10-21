//===-- HLFIRDialect.cpp --------------------------------------------------===//
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

#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flang/Optimizer/HLFIR/HLFIRDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRAttributes.cpp.inc"

void hlfir::hlfirDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flang/Optimizer/HLFIR/HLFIRTypes.cpp.inc"
      >();
}

// `expr` `<` `*` | bounds (`x` bounds)* `:` type [`?`] `>`
// bounds ::= `?` | int-lit
mlir::Type hlfir::ExprType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return {};
  ExprType::Shape shape;
  if (parser.parseOptionalStar()) {
    if (parser.parseDimensionList(shape, /*allowDynamic=*/true))
      return {};
  } else if (parser.parseColon()) {
    return {};
  }
  mlir::Type eleTy;
  if (parser.parseType(eleTy))
    return {};
  const bool polymorphic = mlir::succeeded(parser.parseOptionalQuestion());
  if (parser.parseGreater())
    return {};
  return ExprType::get(parser.getContext(), shape, eleTy, polymorphic);
}

void hlfir::ExprType::print(mlir::AsmPrinter &printer) const {
  auto shape = getShape();
  printer << '<';
  if (shape.size()) {
    for (const auto &b : shape) {
      if (b >= 0)
        printer << b << 'x';
      else
        printer << "?x";
    }
  }
  printer << getEleTy();
  if (isPolymorphic())
    printer << '?';
  printer << '>';
}
