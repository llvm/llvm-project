//===- CIRTypes.cpp - MLIR CIR Types --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsTypes.cpp.inc"

using namespace mlir;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
// General CIR parsing / printing
//===----------------------------------------------------------------------===//

Type CIRDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  Type genType;
  OptionalParseResult parseResult =
      generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  parser.emitError(typeLoc, "unknown type in CIR dialect");
  return Type();
}

void CIRDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (failed(generatedTypePrinter(type, os)))
    llvm_unreachable("unexpected CIR type kind");
}

Type PointerType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  Type pointeeType;
  if (parser.parseType(pointeeType))
    return Type();
  if (parser.parseGreater())
    return Type();
  return get(parser.getContext(), pointeeType);
}

void PointerType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printType(getPointee());
  printer << '>';
}

Type BoolType::parse(mlir::AsmParser &parser) {
  return get(parser.getContext());
}

void BoolType::print(mlir::AsmPrinter &printer) const {}

Type StructType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  std::string typeName;
  if (parser.parseString(&typeName))
    return Type();
  llvm::SmallVector<Type> members;
  Type nextMember;
  while (mlir::succeeded(parser.parseType(nextMember)))
    members.push_back(nextMember);
  if (parser.parseGreater())
    return Type();
  return get(parser.getContext(), members, typeName);
}

void StructType::print(mlir::AsmPrinter &printer) const {
  printer << '<' << getTypeName() << ", ";
  llvm::interleaveComma(getMembers(), printer);
  printer << '>';
}

Type ArrayType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  Type eltType;
  if (parser.parseType(eltType))
    return Type();
  if (parser.parseKeyword("x"))
    return Type();

  uint64_t val = 0;
  if (parser.parseInteger(val).failed())
    return Type();

  if (parser.parseGreater())
    return Type();
  return get(parser.getContext(), eltType, val);
}

void ArrayType::print(mlir::AsmPrinter &printer) const {
  printer << '<';
  printer.printType(getEltType());
  printer << " x " << getSize();
  printer << '>';
}

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//

void CIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "clang/CIR/Dialect/IR/CIROpsTypes.cpp.inc"
      >();
}
