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
  bool parsedBody = false;

  auto parseASTAttribute = [&](Attribute &attr) {
    auto optAttr = parser.parseOptionalAttribute(attr);
    if (optAttr.has_value()) {
      if (failed(*optAttr))
        return false;
      if (attr.isa<ASTFunctionDeclAttr>() || attr.isa<ASTRecordDeclAttr>() ||
          attr.isa<ASTVarDeclAttr>())
        return true;
      parser.emitError(parser.getCurrentLocation(),
                       "Unknown cir.struct attribute");
      return false;
    }
    return false;
  };

  while (mlir::succeeded(parser.parseOptionalComma())) {
    if (mlir::succeeded(parser.parseOptionalKeyword("incomplete")))
      continue;

    parsedBody = true;
    Type nextMember;
    auto optTy = parser.parseOptionalType(nextMember);
    if (optTy.has_value()) {
      if (failed(*optTy))
        return Type();
      members.push_back(nextMember);
      continue;
    }

    // Maybe it's an AST attribute: always last member, break.
    Attribute astAttr;
    if (parseASTAttribute(astAttr))
      break;
  }

  if (parser.parseGreater())
    return Type();
  auto sTy = get(parser.getContext(), members, typeName, parsedBody);
  return sTy;
}

void StructType::print(mlir::AsmPrinter &printer) const {
  printer << '<' << getTypeName();
  if (!getBody()) {
    printer << ", incomplete";
  } else {
    auto members = getMembers();
    if (!members.empty()) {
      printer << ", ";
      llvm::interleaveComma(getMembers(), printer);
    }
  }
  if (getAst()) {
    printer << ", ";
    printer.printAttributeWithoutType(*getAst());
  }
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
